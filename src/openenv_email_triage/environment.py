"""
EmailTriageEnv v2 — OpenEnv-compliant email triage with sequential state.

What makes this a true sequential decision problem (not just N independent
classifications):

  1. ESCALATION BUDGET  — The agent has a fixed number of flag_review=True
     uses per episode. Wasting budget on low-priority emails means critical
     ones cannot be escalated later. Budget is visible in every observation.

  2. TEAM QUEUE CAPACITY — Each routing destination has a finite capacity.
     Routing too many emails to legal/tier2/management saturates the queue;
     subsequent emails routed there incur an overflow penalty and the agent
     must adapt (e.g. route to management instead of legal when legal is full).

  3. SLA DECAY TIMERS — Every email has a deadline relative to when it arrived.
     If the agent processes low-priority emails first and leaves urgent ones
     untouched, SLA breach events fire automatically at the start of each step,
     penalising the agent. Processing order therefore matters.

  4. CRITICAL CASCADE — If 2+ urgent emails breach SLA in the same episode,
     a one-time cascade penalty triggers and the observation marks
     cascade_active=True, signalling compounding organisational damage.

These mechanics mean:
  - Agent must read future inbox to plan processing ORDER (planning horizon).
  - Agent must ration escalations across the full episode (budget constraint).
  - Routing choices affect available routes for later emails (resource constraint).
  - Greedy per-email optimisation is strictly suboptimal.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action, EmailMessage, EmailHeader, EnvironmentState,
    Observation, Reward, RewardBreakdown,
    SessionConstraints, SlaStatus,
    TEAM_CAPACITY, SLA_STEPS,
)
from .dataset import EASY_EMAILS, MEDIUM_EMAILS, HARD_EMAILS
from .grader import score_action, grade_episode

TASK_DATASETS: Dict[str, List[Dict[str, Any]]] = {
    "easy":   EASY_EMAILS,
    "medium": MEDIUM_EMAILS,
    "hard":   HARD_EMAILS,
}

# Per-task escalation budgets (hard is tightest relative to true need)
TASK_ESCALATION_BUDGET: Dict[str, int] = {
    "easy":   3,   # 2 emails truly need escalation → budget=3, comfortable
    "medium": 4,   # 4 emails truly need escalation → budget=4, exact
    "hard":   5,   # 7 emails truly need escalation → budget=5, must choose
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "easy": (
        "Triage 5 emails. Escalation budget=3 (2 truly required). "
        "Team queues are generous. SLA deadlines are forgiving. "
        "Expected score for a competent agent: 0.75–0.90."
    ),
    "medium": (
        "Triage 8 emails. Escalation budget=4 (exactly matching true need). "
        "legal and support_tier2 queues can saturate if misused. "
        "Processing order affects SLA breaches. "
        "Expected score: 0.55–0.75."
    ),
    "hard": (
        "Triage 10 emails. Escalation budget=5 but 7 emails truly require "
        "escalation — agent must choose which 5 matter most. "
        "legal queue capacity=2, management=2; overflow forces creative routing. "
        "Multiple urgent SLA timers run simultaneously. "
        "Expected score: 0.35–0.60."
    ),
}


def _build_email_message(raw: Dict[str, Any]) -> EmailMessage:
    header = EmailHeader(**raw["email"]["header"])
    return EmailMessage(
        header=header,
        body=raw["email"]["body"],
        metadata=raw["email"].get("metadata", {}),
    )


class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage environment with sequential state.
    """

    ENV_ID   = "email-triage-v1"
    VERSION  = "2.0.0"
    MAX_STEPS = 60

    def __init__(self, task_id: str = "easy", seed: Optional[int] = None) -> None:
        if task_id not in TASK_DATASETS:
            raise ValueError(f"task_id must be one of {list(TASK_DATASETS.keys())}")
        self.task_id  = task_id
        self.seed     = seed
        self._dataset = TASK_DATASETS[task_id]
        # Runtime state
        self._emails:         List[EmailMessage]    = []
        self._processed_ids:  List[str]             = []
        self._actions_log:    List[Dict[str, Any]]  = []
        self._step_num:       int                   = 0
        self._done:           bool                  = False
        self._cumulative_reward: float              = 0.0
        self._constraints:    SessionConstraints    = SessionConstraints()

    # ── OpenEnv Interface ─────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset all state and return initial observation."""
        self._emails         = [_build_email_message(e) for e in self._dataset]
        self._processed_ids  = []
        self._actions_log    = []
        self._step_num       = 0
        self._done           = False
        self._cumulative_reward = 0.0

        budget = TASK_ESCALATION_BUDGET[self.task_id]
        self._constraints = SessionConstraints(escalation_budget=budget)

        # Register every email in the SLA tracker immediately
        for i, raw in enumerate(self._dataset):
            gt_priority = raw["ground_truth"]["priority"]
            deadline    = i + SLA_STEPS.get(gt_priority, 99)
            self._constraints.sla_tracker.append(SlaStatus(
                email_id        = raw["email"]["header"]["email_id"],
                true_priority   = gt_priority,
                arrived_at_step = i,     # emails are revealed sequentially
                deadline_step   = deadline,
            ))

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one agent action.

        Sequential effects applied each step (before scoring the action):
          • SLA breach check: any email whose deadline < current step fires a penalty.
          • Cascade check: ≥2 urgent SLA breaches triggers cascade_active.

        Then the action is scored, and sequential resource effects are applied:
          • Escalation budget decremented if flag_review=True.
          • Team queue capacity decremented for route_to destination.
        """
        if self._done:
            raise RuntimeError("Episode done. Call reset() to start a new one.")
        if self._step_num >= self.MAX_STEPS:
            self._done = True
            return (
                self._make_observation(),
                Reward(total=0.0, done=True, info={"reason": "max_steps_exceeded"}),
                True, {"reason": "max_steps_exceeded"},
            )

        breakdown = RewardBreakdown()

        # ── A. SLA decay: fire breach penalties for unprocessed overdue emails ──
        sla_penalty = self._tick_sla(breakdown)

        # ── B. Validate email_id ──────────────────────────────────────────────
        inbox_ids = {e.header.email_id for e in self._emails}
        if action.email_id not in inbox_ids:
            breakdown.base_penalty = -0.1
            total = max(-1.0, sla_penalty - 0.1)
            r = Reward(total=total, breakdown=breakdown, done=False,
                       info={"error": f"email_id '{action.email_id}' not in inbox"})
            self._step_num += 1
            self._cumulative_reward += r.total
            return self._make_observation(), r, False, r.info

        # ── C. Score the classification decision (label correctness) ──────────
        label_reward, detail = score_action(action)
        breakdown.priority_score   = label_reward.breakdown.priority_score
        breakdown.category_score   = label_reward.breakdown.category_score
        breakdown.routing_score    = label_reward.breakdown.routing_score
        breakdown.summary_score    = label_reward.breakdown.summary_score
        breakdown.escalation_score = label_reward.breakdown.escalation_score
        breakdown.base_penalty     = label_reward.breakdown.base_penalty

        # ── D. Sequential resource effects ────────────────────────────────────

        # D1. Escalation budget
        budget_penalty = 0.0
        if action.flag_review:
            if self._constraints.escalations_used >= self._constraints.escalation_budget:
                # Went over budget — escalation is silently dropped and penalised
                budget_penalty = -0.20
                breakdown.budget_penalty = budget_penalty
                # Force flag_review=False for grading purposes (budget exhausted)
                detail["budget_overflow"] = True
            else:
                self._constraints.escalations_used += 1

        # D2. Team queue capacity
        queue_penalty = 0.0
        route_key = action.route_to.value
        if route_key not in ("trash", "archive"):
            accepted = self._constraints.team_queues.consume(route_key)
            if not accepted:
                queue_penalty = -0.10
                breakdown.queue_penalty = queue_penalty
                self._constraints.queue_overflows += 1
                detail["queue_overflow"] = route_key

        # ── E. Mark email as processed; remove from inbox ─────────────────────
        self._actions_log.append(action.model_dump())
        self._processed_ids.append(action.email_id)
        self._emails = [e for e in self._emails if e.header.email_id != action.email_id]

        # Mark SLA entry as handled (no further breach risk)
        for sla in self._constraints.sla_tracker:
            if sla.email_id == action.email_id:
                sla.breached = True   # "handled" — no future breach fires
                break

        # ── F. Cascade check ──────────────────────────────────────────────────
        cascade_penalty = 0.0
        urgent_breaches = sum(
            1 for s in self._constraints.sla_tracker
            if s.breached and s.true_priority == "urgent"
               and s.deadline_step <= self._step_num   # breached late, not just handled
        )
        if urgent_breaches >= 2 and not self._constraints.cascade_triggered:
            self._constraints.cascade_triggered = True
            cascade_penalty = -0.25
            breakdown.cascade_penalty = cascade_penalty

        # ── G. Compute total reward ───────────────────────────────────────────
        label_total = label_reward.total   # already clamped [0,1]
        sequential_penalties = sla_penalty + budget_penalty + queue_penalty + cascade_penalty
        total = max(-1.0, min(1.0, label_total + sequential_penalties))

        breakdown.sla_penalty = sla_penalty

        self._step_num += 1
        self._cumulative_reward += total
        self._done = len(self._emails) == 0

        detail.update({
            "sla_penalty":     sla_penalty,
            "queue_penalty":   queue_penalty,
            "budget_penalty":  budget_penalty,
            "cascade_penalty": cascade_penalty,
            "escalations_remaining": (
                self._constraints.escalation_budget - self._constraints.escalations_used
            ),
        })

        reward = Reward(total=round(total, 4), breakdown=breakdown,
                        done=self._done, info=detail)

        if self._done:
            reward.info["episode_summary"] = grade_episode(self._actions_log)
            reward.info["final_constraints"] = self._constraints_dict()

        return self._make_observation(), reward, self._done, reward.info

    def state(self) -> EnvironmentState:
        """Full internal state snapshot (for logging / debugging)."""
        scores = {}
        if self._actions_log:
            s = grade_episode(self._actions_log)
            scores = {"running_score": s["overall_score"], "emails_triaged": s["num_emails"]}
        return EnvironmentState(
            task_id           = self.task_id,
            step              = self._step_num,
            done              = self._done,
            observation       = self._make_observation(),
            cumulative_reward = round(self._cumulative_reward, 4),
            actions_taken     = copy.deepcopy(self._actions_log),
            grader_scores     = scores,
            constraints       = self._constraints_dict(),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _tick_sla(self, breakdown: RewardBreakdown) -> float:
        """
        Check for SLA breaches among unprocessed emails.
        A breach fires the first time step_num > deadline_step.
        Returns total SLA penalty for this step.
        """
        penalty = 0.0
        processed_set = set(self._processed_ids)
        for sla in self._constraints.sla_tracker:
            if sla.email_id in processed_set:
                continue   # already handled
            if sla.breached:
                continue   # already penalised
            if self._step_num >= sla.deadline_step:
                sla.breached = True   # mark as breached (not just handled)
                self._constraints.sla_breaches += 1
                penalty -= 0.15
        return penalty

    def _make_observation(self) -> Observation:
        c = self._constraints
        # Build SLA warnings for emails still in inbox
        processed_set = set(self._processed_ids)
        warnings = []
        for sla in c.sla_tracker:
            if sla.email_id in processed_set:
                continue
            steps_left = sla.deadline_step - self._step_num
            if steps_left <= 2 and not sla.breached:
                warnings.append({
                    "email_id":    sla.email_id,
                    "priority":    sla.true_priority,
                    "steps_left":  max(0, steps_left),
                    "overdue":     steps_left < 0,
                })

        from .models import Priority, Category, RouteTo
        queue_dict = {
            k: c.team_queues.remaining(k)
            for k in TEAM_CAPACITY
        }

        return Observation(
            inbox         = list(self._emails),
            processed     = list(self._processed_ids),
            current_email = self._emails[0] if self._emails else None,
            step_number   = self._step_num,
            total_emails  = len(self._dataset),
            remaining     = len(self._emails),
            escalation_budget_remaining = (
                c.escalation_budget - c.escalations_used
            ),
            team_queue_remaining = queue_dict,
            active_sla_warnings  = warnings,
            sla_breaches_so_far  = c.sla_breaches,
            cascade_active       = c.cascade_triggered,
            session_info = {
                "task_id":          self.task_id,
                "task_description": TASK_DESCRIPTIONS[self.task_id],
                "cumulative_reward": round(self._cumulative_reward, 4),
                "action_space": {
                    "priority":    [p.value for p in Priority],
                    "category":    [c2.value for c2 in Category],
                    "route_to":    [r.value for r in RouteTo],
                    "summary":     "string ≤280 chars",
                    "flag_review": "bool — uses escalation budget",
                    "reasoning":   "string, not scored",
                },
                "constraints_info": {
                    "escalation_budget": c.escalation_budget,
                    "escalations_used":  c.escalations_used,
                    "sla_breaches":      c.sla_breaches,
                    "queue_overflows":   c.queue_overflows,
                    "cascade_triggered": c.cascade_triggered,
                },
            },
        )

    def _constraints_dict(self) -> Dict[str, Any]:
        c = self._constraints
        return {
            "escalation_budget":     c.escalation_budget,
            "escalations_used":      c.escalations_used,
            "escalations_remaining": c.escalation_budget - c.escalations_used,
            "sla_breaches":          c.sla_breaches,
            "queue_overflows":       c.queue_overflows,
            "cascade_triggered":     c.cascade_triggered,
            "team_queues":           {k: c.team_queues.remaining(k) for k in TEAM_CAPACITY},
        }

    @property
    def is_done(self) -> bool:
        return self._done

    def __repr__(self) -> str:
        c = self._constraints
        return (
            f"EmailTriageEnv(task={self.task_id}, step={self._step_num}, "
            f"budget_left={c.escalation_budget - c.escalations_used}, "
            f"sla_breaches={c.sla_breaches}, done={self._done})"
        )
