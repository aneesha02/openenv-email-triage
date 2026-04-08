#!/usr/bin/env python3
"""
inference.py — OpenEnv Hackathon Submission
Email Triage Environment

STDOUT FORMAT (mandatory):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Add src to path so we can import the environment ──────────────────────────


from environment import EmailTriageEnv
from models import Priority, Category, RouteTo, Action

# ── Required env vars ─────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")


TASKS        = ["easy", "medium", "hard"]
BENCHMARK    = "email-triage-v1"
SUCCESS_THRESHOLD = 0.5   # score >= 0.5 counts as success

# ── Logging helpers (exact format required by hackathon) ──────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage assistant for a B2B SaaS company.
You must classify each email and return ONLY a valid JSON object.

Return exactly this structure (no markdown, no explanation, just JSON):
{
  "email_id": "<copy exactly from the email>",
  "priority": "<urgent|high|medium|low|spam>",
  "category": "<customer_complaint|billing_inquiry|technical_support|sales_lead|internal_hr|legal_compliance|spam_phishing|general_inquiry>",
  "route_to": "<support_tier1|support_tier2|billing|sales|legal|hr|management|trash|archive>",
  "summary": "<max 280 chars concise summary>",
  "flag_review": <true|false>,
  "reasoning": "<brief reasoning>"
}

Priority rules:
- urgent: legal deadline <72h, security incident, production outage, regulatory action
- high: legal threat, important sales, billing dispute, confidential HR, board matters
- medium: routine billing question, general inquiry, standard support
- low: internal social, scheduling, low-priority updates
- spam: phishing, scams, unsolicited commercial email

Routing rules:
- support_tier1: simple password resets, basic how-to
- support_tier2: production outages, security incidents, complex technical issues
- billing: invoices, subscriptions, payment failures
- sales: new business, enterprise leads, partnerships
- legal: regulatory notices, legal threats, compliance, contracts
- hr: employee relations, PIP, hiring, misconduct
- management: exec decisions, acquisition, crisis, major SLA breach
- trash: spam, phishing — delete immediately
- archive: low-priority non-actionable

IMPORTANT sequential constraints visible in the observation:
- escalation_budget_remaining: only flag_review=true if budget > 0 and email truly needs escalation
- team_queue_remaining: avoid routing to teams with 0 remaining capacity
- active_sla_warnings: process emails with steps_left=0 or 1 FIRST"""


def build_user_prompt(obs: dict) -> str:
    """Build the prompt for the current step from the observation dict."""
    current = obs.get("current_email")
    if not current:
        return "No email to process."

    header = current["header"]
    body   = current["body"]

    budget   = obs.get("escalation_budget_remaining", 0)
    queues   = obs.get("team_queue_remaining", {})
    warnings = obs.get("active_sla_warnings", [])
    step     = obs.get("step_number", 0)
    remaining = obs.get("remaining", 0)

    sla_note = ""
    if warnings:
        sla_note = f"\n⚠️  SLA WARNINGS: {json.dumps(warnings)}"

    full_queues = [k for k, v in queues.items() if v == 0]
    queue_note = f"\n🚫 FULL QUEUES (do not route here): {full_queues}" if full_queues else ""

    return f"""Step {step} — {remaining} emails remaining.
Escalation budget left: {budget}{sla_note}{queue_note}

EMAIL TO TRIAGE:
ID:      {header['email_id']}
From:    {header['sender']}
Subject: {header['subject']}
Time:    {header['timestamp']}

{body}

Return ONLY a JSON action object."""


def parse_action(raw: str, fallback_email_id: str) -> Dict[str, Any]:
    """Parse the model's JSON response into an action dict."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        data = json.loads(text)
        # Ensure required fields exist with safe defaults
        return {
            "email_id":    data.get("email_id", fallback_email_id),
            "priority":    data.get("priority", "medium"),
            "category":    data.get("category", "general_inquiry"),
            "route_to":    data.get("route_to", "support_tier1"),
            "summary":     str(data.get("summary", "Email processed."))[:280],
            "flag_review": bool(data.get("flag_review", False)),
            "reasoning":   str(data.get("reasoning", "")),
        }
    except Exception:
        # Fallback: safe default action
        return {
            "email_id":    fallback_email_id,
            "priority":    "medium",
            "category":    "general_inquiry",
            "route_to":    "support_tier1",
            "summary":     "Unable to parse model response — defaulting to general inquiry.",
            "flag_review": False,
            "reasoning":   f"Parse error on: {raw[:100]}",
        }


def rule_based_action(obs: dict) -> Dict[str, Any]:
    """Fallback rule-based agent when no API key is set."""
    current = obs.get("current_email")
    if not current:
        return {}
    header = current["header"]
    body   = current["body"]
    email_id = header["email_id"]
    subject  = (header["subject"] + " " + body).lower()

    budget = obs.get("escalation_budget_remaining", 0)
    queues = obs.get("team_queue_remaining", {})

    priority    = "medium"
    category    = "general_inquiry"
    route_to    = "support_tier1"
    flag_review = False

    if any(w in subject for w in ["spam", "phishing", "congratulations", "won $", "lottery", "verify your"]):
        priority = "spam"; category = "spam_phishing"; route_to = "trash"
    elif any(w in subject for w in ["legal", "lawsuit", "compliance", "regulation", "gdpr", "breach notice"]):
        priority = "urgent"; category = "legal_compliance"
        route_to = "legal" if queues.get("legal", 0) > 0 else "management"
        flag_review = budget > 0 and priority in ["urgent", "high"]
    elif any(w in subject for w in ["hacked", "outage", "security", "ransomware", "incident"]):
        priority = "urgent"; category = "technical_support"; route_to = "support_tier2"
        flag_review = budget > 0 and priority in ["urgent", "high"]
    elif any(w in subject for w in ["invoice", "billing", "payment", "subscription", "overdue"]):
        priority = "high"; category = "billing_inquiry"; route_to = "billing"
    elif any(w in subject for w in ["enterprise", "sales", "pricing", "license", "acquisition"]):
        priority = "high"; category = "sales_lead"; route_to = "sales"
    elif any(w in subject for w in ["hr", "pip", "performance", "misconduct", "termination", "wages"]):
        priority = "high"; category = "internal_hr"
        route_to = "hr" if queues.get("hr", 0) > 0 else "management"
        flag_review = budget > 0 and priority in ["urgent", "high"]

    return {
        "email_id":    email_id,
        "priority":    priority,
        "category":    category,
        "route_to":    route_to,
        "summary":     header["subject"][:280],
        "flag_review": flag_review,
        "reasoning":   "Rule-based heuristic",
    }


def run_task(client: Optional[OpenAI], task_id: str) -> float:
    """Run one task episode and return the final score."""
    env = EmailTriageEnv(task_id=task_id, seed=42)
    obs_obj = env.reset()
    obs = obs_obj.model_dump()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME if client else "rule-based-demo")

    try:
        step = 0
        while not env.is_done:
            step += 1
            current = obs.get("current_email")
            if not current:
                break

            fallback_id = current["header"]["email_id"]
            error_msg = None

            # Get action from model or rule-based fallback
            if client:
                try:
                    user_prompt = build_user_prompt(obs)
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": user_prompt},
                        ],
                        temperature=0.2,
                        max_tokens=400,
                    )
                    raw = (completion.choices[0].message.content or "").strip()
                    action_dict = parse_action(raw, fallback_id)
                except Exception as exc:
                    error_msg = str(exc)[:80]
                    action_dict = rule_based_action(obs)
            else:
                action_dict = rule_based_action(obs)

            # Build and validate Action
            try:
                action = Action(
                    email_id    = action_dict["email_id"],
                    priority    = Priority(action_dict["priority"]),
                    category    = Category(action_dict["category"]),
                    route_to    = RouteTo(action_dict["route_to"]),
                    summary     = action_dict["summary"],
                    flag_review = action_dict["flag_review"],
                    reasoning   = action_dict.get("reasoning", ""),
                )
            except Exception as exc:
                error_msg = f"invalid_action:{exc}"
                # Safe fallback action
                action = Action(
                    email_id    = fallback_id,
                    priority    = Priority.MEDIUM,
                    category    = Category.GENERAL_INQUIRY,
                    route_to    = RouteTo.SUPPORT_TIER1,
                    summary     = "Fallback — action validation failed.",
                    flag_review = False,
                )

            obs_obj, reward_obj, done, info = env.step(action)
            obs = obs_obj.model_dump()

            reward = reward_obj.total
            rewards.append(reward)
            steps_taken = step

            action_str = (
                f"{{id={action.email_id},pri={action.priority.value},"
                f"cat={action.category.value},route={action.route_to.value},"
                f"flag={action.flag_review}}}"
            )
            env_error = None
            if isinstance(info, dict):
                env_error = info.get("last_action_error") or info.get("error")

            log_step(step=step, action=action_str, reward=reward,
                    done=done, error=env_error)

        # Compute final score from grader
        from grader import grade_episode
        grader_result = grade_episode(env._actions_log)
        score = grader_result.get("label_score", 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)

    finally:
        try:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

def run():
    """
    OpenEnv entrypoint (called by server)
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    score = run_task(client, "easy")  # single task only

    return {
        "status": "success",
        "score": score
    }

def main() -> None:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is not set")
    client: Optional[OpenAI] = None
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_scores: Dict[str, float] = {}
    for task_id in TASKS:
        score = run_task(client, task_id)
        all_scores[task_id] = score

    # Summary to stderr so it doesn't pollute the required stdout format
    print("\n=== FINAL SCORES ===", file=sys.stderr)
    for task_id, score in all_scores.items():
        print(f"  {task_id:<8} {score:.4f}", file=sys.stderr)
    overall = sum(all_scores.values()) / len(all_scores)
    print(f"  OVERALL  {overall:.4f}", file=sys.stderr)

    # Write results JSON
    results_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model":       MODEL_NAME if client else "rule-based-demo",
            "benchmark":   BENCHMARK,
            "tasks":       all_scores,
            "overall":     round(overall, 4),
        }, f, indent=2)
    print(f"\nResults written to {results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
