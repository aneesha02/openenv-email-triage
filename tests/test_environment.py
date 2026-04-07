"""
pytest suite for Email Triage OpenEnv v2.
Covers: model validation, grader, environment lifecycle,
sequential mechanics (SLA/budget/queue/cascade), and full episodes.

Run: pytest tests/ -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openenv_email_triage import (
    EmailTriageEnv, Action, Priority, Category, RouteTo
)
from openenv_email_triage.grader import score_action, grade_episode
from openenv_email_triage.models import (
    Observation, Reward, EnvironmentState,
    TEAM_CAPACITY, SLA_STEPS, TASK_ESCALATION_BUDGET,
)
from openenv_email_triage.dataset import EASY_EMAILS, MEDIUM_EMAILS, HARD_EMAILS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    e = EmailTriageEnv(task_id="easy"); e.reset(); return e

@pytest.fixture
def hard_env():
    e = EmailTriageEnv(task_id="hard"); e.reset(); return e

GT = {
    d["email"]["header"]["email_id"]: d["ground_truth"]
    for dataset in [EASY_EMAILS, MEDIUM_EMAILS, HARD_EMAILS]
    for d in dataset
}

def perfect(email_id: str) -> Action:
    gt = GT[email_id]
    return Action(
        email_id    = email_id,
        priority    = Priority(gt["priority"]),
        category    = Category(gt["category"]),
        route_to    = RouteTo(gt["route_to"]),
        summary     = f"Triaged {email_id}: {gt['category']} → {gt['route_to']}",
        flag_review = gt.get("requires_escalation", False),
    )


# ── 1. Model layer ────────────────────────────────────────────────────────────

def test_action_valid():
    a = Action(
        email_id="e001", priority=Priority.URGENT,
        category=Category.TECHNICAL_SUPPORT, route_to=RouteTo.SUPPORT_TIER2,
        summary="Account compromised — urgent lockdown needed.", flag_review=True,
    )
    assert a.priority == Priority.URGENT
    assert a.flag_review is True

def test_action_summary_max_length():
    with pytest.raises(Exception):
        Action(
            email_id="e001", priority=Priority.LOW,
            category=Category.GENERAL_INQUIRY, route_to=RouteTo.SUPPORT_TIER1,
            summary="x" * 281, flag_review=False,
        )

def test_observation_has_sequential_fields(easy_env):
    obs = easy_env._make_observation()
    assert hasattr(obs, "escalation_budget_remaining")
    assert hasattr(obs, "team_queue_remaining")
    assert hasattr(obs, "active_sla_warnings")
    assert hasattr(obs, "sla_breaches_so_far")
    assert hasattr(obs, "cascade_active")

def test_reward_has_sequential_breakdown_fields(easy_env):
    obs = easy_env._make_observation()
    eid = obs.current_email.header.email_id
    _, r, _, _ = easy_env.step(perfect(eid))
    assert hasattr(r.breakdown, "sla_penalty")
    assert hasattr(r.breakdown, "queue_penalty")
    assert hasattr(r.breakdown, "budget_penalty")
    assert hasattr(r.breakdown, "cascade_penalty")


# ── 2. Grader ─────────────────────────────────────────────────────────────────

def test_perfect_action_high_score():
    reward, _ = score_action(perfect("e001"))
    assert reward.total >= 0.85

def test_wrong_priority_lowers_score():
    correct, _ = score_action(perfect("e001"))
    wrong = Action(
        email_id="e001", priority=Priority.LOW,
        category=Category.TECHNICAL_SUPPORT, route_to=RouteTo.SUPPORT_TIER2,
        summary="Test.", flag_review=True,
    )
    wrong_r, _ = score_action(wrong)
    assert wrong_r.total < correct.total

def test_spam_correctly_identified():
    r, _ = score_action(perfect("e002"))
    assert r.total >= 0.85

def test_spam_misrouted_penalised():
    a = Action(
        email_id="e002", priority=Priority.SPAM,
        category=Category.SPAM_PHISHING, route_to=RouteTo.SALES,
        summary="Spam.", flag_review=False,
    )
    _, detail = score_action(a)
    assert detail["scores"]["base_penalty"] < 0

def test_urgent_as_spam_heavy_penalty():
    a = Action(
        email_id="e001", priority=Priority.SPAM,
        category=Category.SPAM_PHISHING, route_to=RouteTo.TRASH,
        summary="Spam.", flag_review=False,
    )
    _, detail = score_action(a)
    assert detail["scores"]["base_penalty"] <= -0.3

def test_adjacent_priority_partial_credit():
    a = Action(
        email_id="e001", priority=Priority.HIGH,   # adjacent to urgent
        category=Category.TECHNICAL_SUPPORT, route_to=RouteTo.SUPPORT_TIER2,
        summary="Security issue.", flag_review=True,
    )
    _, detail = score_action(a)
    assert 0 < detail["scores"]["priority"] < 1

def test_missed_escalation_zero_score():
    a = Action(
        email_id="e001", priority=Priority.URGENT,
        category=Category.TECHNICAL_SUPPORT, route_to=RouteTo.SUPPORT_TIER2,
        summary="Security.", flag_review=False,   # should be True
    )
    _, detail = score_action(a)
    assert detail["scores"]["escalation"] == 0.0

def test_over_escalation_half_score():
    a = Action(
        email_id="e003", priority=Priority.LOW,
        category=Category.INTERNAL_HR, route_to=RouteTo.HR,
        summary="Team lunch reminder.", flag_review=True,   # not needed
    )
    _, detail = score_action(a)
    assert detail["scores"]["escalation"] == 0.5

def test_grade_episode_aggregation():
    actions = [
        {"email_id":"e001","priority":"urgent","category":"technical_support",
         "route_to":"support_tier2","summary":"Security incident.",
         "flag_review":True,"reasoning":""},
        {"email_id":"e002","priority":"spam","category":"spam_phishing",
         "route_to":"trash","summary":"Spam.","flag_review":False,"reasoning":""},
    ]
    result = grade_episode(actions)
    assert 0.0 <= result["label_score"] <= 1.0
    assert result["num_emails"] == 2


# ── 3. Environment lifecycle ──────────────────────────────────────────────────

def test_reset_returns_observation(easy_env):
    obs = easy_env.reset()
    assert isinstance(obs, Observation)
    assert obs.total_emails == 5
    assert obs.remaining == 5
    assert obs.current_email is not None

def test_reset_initialises_sequential_state(easy_env):
    obs = easy_env.reset()
    assert obs.escalation_budget_remaining == TASK_ESCALATION_BUDGET["easy"]
    assert obs.sla_breaches_so_far == 0
    assert not obs.cascade_active
    for k, cap in TEAM_CAPACITY.items():
        assert obs.team_queue_remaining[k] == cap

def test_step_types(easy_env):
    eid = easy_env._make_observation().current_email.header.email_id
    obs, reward, done, info = easy_env.step(perfect(eid))
    assert isinstance(obs, Observation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_step_removes_email_from_inbox(easy_env):
    obs = easy_env._make_observation()
    eid = obs.current_email.header.email_id
    obs2, _, _, _ = easy_env.step(perfect(eid))
    assert obs2.remaining == obs.remaining - 1
    assert eid not in [e.header.email_id for e in obs2.inbox]

def test_step_decrements_queue(easy_env):
    obs = easy_env._make_observation()
    eid = obs.current_email.header.email_id   # e001 → support_tier2
    obs2, _, _, _ = easy_env.step(perfect(eid))
    assert obs2.team_queue_remaining["support_tier2"] == TEAM_CAPACITY["support_tier2"] - 1

def test_step_decrements_escalation_budget(easy_env):
    # e001 requires escalation
    obs = easy_env._make_observation()
    eid = obs.current_email.header.email_id
    budget_before = obs.escalation_budget_remaining
    obs2, _, _, _ = easy_env.step(perfect(eid))
    # perfect(e001) has flag_review=True
    assert obs2.escalation_budget_remaining == budget_before - 1

def test_invalid_email_id_penalised(easy_env):
    a = Action(
        email_id="INVALID", priority=Priority.LOW,
        category=Category.GENERAL_INQUIRY, route_to=RouteTo.SUPPORT_TIER1,
        summary="Test.", flag_review=False,
    )
    _, reward, _, _ = easy_env.step(a)
    assert reward.total < 0

def test_state_returns_correct_type(easy_env):
    st = easy_env.state()
    assert isinstance(st, EnvironmentState)
    assert st.task_id == "easy"
    assert not st.done
    assert "escalation_budget" in st.constraints


# ── 4. Sequential mechanics ───────────────────────────────────────────────────

@pytest.mark.parametrize("task_id,n", [("easy",5),("medium",8),("hard",10)])
def test_task_email_count(task_id, n):
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    assert obs.total_emails == n

@pytest.mark.parametrize("task_id", ["easy","medium","hard"])
def test_task_escalation_budget(task_id):
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    assert obs.escalation_budget_remaining == TASK_ESCALATION_BUDGET[task_id]

def test_sla_breach_fires_for_delayed_urgent():
    """Leaving an urgent email untouched for 3 steps triggers SLA breach."""
    env = EmailTriageEnv("easy"); env.reset()
    # e001 (urgent) deadline = arrived_at_step(0) + SLA_STEPS["urgent"](2) = 2
    # Breach fires when self._step >= 2, which happens at the START of step index 2 (3rd action)
    non_urgent = [
        d for d in EASY_EMAILS if d["ground_truth"]["priority"] != "urgent"
    ]
    last_reward = None
    for d in non_urgent[:3]:   # 3 actions → tick at step=2 → breach
        eid = d["email"]["header"]["email_id"]
        _, last_reward, _, _ = env.step(perfect(eid))
    assert env._constraints.sla_breaches >= 1
    assert last_reward.breakdown.sla_penalty < 0

def test_sla_penalty_magnitude():
    env = EmailTriageEnv("easy"); env.reset()
    non_urgent = [d for d in EASY_EMAILS if d["ground_truth"]["priority"] != "urgent"]
    for d in non_urgent[:3]:
        eid = d["email"]["header"]["email_id"]
        _, r, _, _ = env.step(perfect(eid))
    assert r.breakdown.sla_penalty == pytest.approx(-0.15, abs=0.001)

def test_budget_exhaustion_penalty():
    """Escalating beyond the budget incurs -0.20 penalty."""
    env = EmailTriageEnv("easy"); env.reset()   # budget=3
    budget_penalties = []
    for d in EASY_EMAILS:
        eid = d["email"]["header"]["email_id"]
        gt  = d["ground_truth"]
        # Force flag_review=True for every email regardless of need
        a = Action(
            email_id=eid, priority=Priority(gt["priority"]),
            category=Category(gt["category"]), route_to=RouteTo(gt["route_to"]),
            summary=f"Forced escalation of {eid}.", flag_review=True,
        )
        _, r, _, info = env.step(a)
        budget_penalties.append(r.breakdown.budget_penalty)
    # Budget=3, 5 emails all escalated → emails 4 and 5 overflow
    assert sum(1 for p in budget_penalties if p == -0.20) >= 2

def test_queue_saturation_penalty(hard_env):
    """Routing 3+ emails to legal (capacity=2) triggers overflow penalty."""
    legal_emails = [
        d for d in HARD_EMAILS if d["ground_truth"]["route_to"] == "legal"
    ]
    assert len(legal_emails) >= 3, "Need ≥3 legal-routed emails"
    queue_penalties = []
    for d in legal_emails:
        eid = d["email"]["header"]["email_id"]
        gt  = d["ground_truth"]
        # Force routing to legal regardless of capacity
        a = Action(
            email_id=eid, priority=Priority(gt["priority"]),
            category=Category(gt["category"]), route_to=RouteTo.LEGAL,
            summary=f"Routing {eid} to legal.", flag_review=False,
        )
        _, r, _, _ = hard_env.step(a)
        queue_penalties.append(r.breakdown.queue_penalty)
    assert any(p == -0.10 for p in queue_penalties)
    assert hard_env._constraints.queue_overflows >= 1

def test_cascade_triggers_after_two_urgent_sla_breaches():
    """Processing urgents in wrong order causes cascade."""
    env = EmailTriageEnv("hard"); env.reset()
    # h001 deadline=2, h002 deadline=3
    non_urgent = [d for d in HARD_EMAILS if d["ground_truth"]["priority"] != "urgent"]
    urgents    = [d for d in HARD_EMAILS if d["ground_truth"]["priority"] == "urgent"]
    for d in non_urgent:                        # steps 0, 1
        env.step(perfect(d["email"]["header"]["email_id"]))
    env.step(perfect(urgents[0]["email"]["header"]["email_id"]))  # step 2: h001 SLA fires
    _, r, _, _ = env.step(perfect(urgents[1]["email"]["header"]["email_id"]))  # step 3: h002 SLA fires
    assert env._constraints.sla_breaches >= 2
    assert env._constraints.cascade_triggered
    assert r.breakdown.cascade_penalty == pytest.approx(-0.25, abs=0.001)

def test_cascade_fires_only_once():
    """Cascade penalty is one-time even if more urgents breach."""
    env = EmailTriageEnv("hard"); env.reset()
    non_urgent = [d for d in HARD_EMAILS if d["ground_truth"]["priority"] != "urgent"]
    urgents    = [d for d in HARD_EMAILS if d["ground_truth"]["priority"] == "urgent"]
    for d in non_urgent:
        env.step(perfect(d["email"]["header"]["email_id"]))
    cascade_penalties = []
    for d in urgents[:4]:
        _, r, _, _ = env.step(perfect(d["email"]["header"]["email_id"]))
        cascade_penalties.append(r.breakdown.cascade_penalty)
    assert sum(1 for p in cascade_penalties if p == -0.25) == 1   # fires exactly once


# ── 5. Full episodes ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", ["easy","medium","hard"])
def test_oracle_episode_completes(task_id):
    """Perfect oracle agent completes episode with all rewards in [−1, 1]."""
    datasets = {"easy": EASY_EMAILS, "medium": MEDIUM_EMAILS, "hard": HARD_EMAILS}
    env = EmailTriageEnv(task_id=task_id); env.reset()
    for d in datasets[task_id]:
        eid = d["email"]["header"]["email_id"]
        _, r, _, _ = env.step(perfect(eid))
        assert -1.0 <= r.total <= 1.0
    assert env.is_done

@pytest.mark.parametrize("task_id", ["easy","medium","hard"])
def test_oracle_label_scores_above_floor(task_id):
    """Oracle label scores should be high (before sequential penalties)."""
    from openenv_email_triage.grader import grade_episode
    datasets = {"easy": EASY_EMAILS, "medium": MEDIUM_EMAILS, "hard": HARD_EMAILS}
    env = EmailTriageEnv(task_id=task_id); env.reset()
    actions = []
    for d in datasets[task_id]:
        eid = d["email"]["header"]["email_id"]
        a = perfect(eid)
        env.step(a)
        actions.append(a.model_dump())
    result = grade_episode(actions)
    assert result["label_score"] >= 0.75, (
        f"Oracle label score too low on {task_id}: {result['label_score']}"
    )

def test_done_raises_on_further_step():
    """Calling step() after episode is done raises RuntimeError."""
    env = EmailTriageEnv("easy"); env.reset()
    for d in EASY_EMAILS:
        env.step(perfect(d["email"]["header"]["email_id"]))
    assert env.is_done
    with pytest.raises(RuntimeError):
        env.step(perfect("e001"))

def test_invalid_task_raises():
    with pytest.raises(ValueError):
        EmailTriageEnv(task_id="impossible")

def test_episode_info_contains_summary_on_done():
    env = EmailTriageEnv("easy"); env.reset()
    last_info = None
    for d in EASY_EMAILS:
        _, _, done, info = env.step(perfect(d["email"]["header"]["email_id"]))
        last_info = info
    assert "episode_summary" in last_info
    assert "label_score" in last_info["episode_summary"]

def test_state_constraints_exposed():
    env = EmailTriageEnv("hard"); env.reset()
    st = env.state()
    c = st.constraints
    assert "escalation_budget" in c
    assert "escalations_used" in c
    assert "sla_breaches" in c
    assert "queue_overflows" in c
    assert "team_queues" in c
