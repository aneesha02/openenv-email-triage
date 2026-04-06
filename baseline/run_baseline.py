#!/usr/bin/env python3
"""
Baseline inference script for the Email Triage OpenEnv.

Runs a language model (via OpenAI-compatible API) against all three task
difficulty levels and reports reproducible scores.

Usage:
    export OPENAI_API_KEY="sk-..."
    export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional
    export OPENAI_MODEL="gpt-4o-mini"                    # optional
    python baseline/run_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Add src to path if running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openenv_email_triage import EmailTriageEnv, Action
from openenv_email_triage.models import Priority, Category, RouteTo
from openenv_email_triage.grader import grade_episode

# ─── Config ──────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL    = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
TASKS    = ["easy", "medium", "hard"]
SEED     = 42

if not API_KEY:
    print("WARNING: OPENAI_API_KEY not set. Set it to run against a real model.")
    print("         Running in demo mode with a rule-based baseline.\n")

client = OpenAI(api_key=API_KEY or "demo", base_url=BASE_URL) if API_KEY else None

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email triage assistant for a B2B SaaS company.
Your job is to classify each email and return a JSON action object.

You must return ONLY a valid JSON object with these exact fields:
{
  "email_id": "<string — copy exactly from the email>",
  "priority": "<urgent|high|medium|low|spam>",
  "category": "<customer_complaint|billing_inquiry|technical_support|sales_lead|internal_hr|legal_compliance|spam_phishing|general_inquiry>",
  "route_to": "<support_tier1|support_tier2|billing|sales|legal|hr|management|trash|archive>",
  "summary": "<max 280 chars — concise triage summary>",
  "flag_review": <true|false — true if escalation to human reviewer is needed>,
  "reasoning": "<brief chain of thought>"
}

Priority guidelines:
- urgent: legal deadline < 72h, security incident, production outage, regulatory action
- high: legal threat, important sales, billing issue, confidential HR, board-level matter
- medium: routine billing question, general customer inquiry
- low: internal social, routine scheduling
- spam: phishing, scams, unsolicited commercial email

Routing guidelines:
- support_tier1: simple password resets, basic how-to questions
- support_tier2: production outages, security incidents, complex technical issues
- billing: invoice disputes, subscription issues, payment failures
- sales: new business inquiries, enterprise leads, partnership discussions
- legal: regulatory notices, legal threats, compliance, contracts with legal terms
- hr: employee relations, PIP, hiring, internal misconduct
- management: exec-level decisions, acquisition, crisis, major SLA breach
- trash: spam, phishing — DELETE immediately
- archive: low-priority non-actionable

flag_review = true when: legal action threatened, regulatory deadline, exec-level decision needed,
  security incident, whistleblower report, media inquiry, acquisition discussion,
  sophisticated phishing targeting organization.

Return ONLY the JSON object. No markdown, no explanation outside the JSON."""


def build_user_prompt(email_id: str, sender: str, subject: str, body: str,
                      has_attachment: bool, timestamp: str) -> str:
    attach = "Yes" if has_attachment else "No"
    return f"""Triage this email:

EMAIL ID: {email_id}
FROM: {sender}
SUBJECT: {subject}
DATE: {timestamp}
HAS ATTACHMENT: {attach}

BODY:
{body}

Return ONLY the JSON action object."""


def call_model(prompt: str, retries: int = 3) -> Optional[Dict[str, Any]]:
    """Call the LLM and parse the JSON response."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=400,
                seed=SEED,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [JSON parse error attempt {attempt+1}]: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"  [API error attempt {attempt+1}]: {e}")
            time.sleep(2)
    return None


# ─── Rule-based demo baseline (used when no API key) ─────────────────────────
SPAM_KEYWORDS = {"lottery", "won", "winner", "click", "verify", "paypal", "microsoft365.cloud"}

def rule_based_triage(email_id: str, sender: str, subject: str, body: str) -> Dict[str, Any]:
    """Simple heuristic agent for demo / CI purposes."""
    text = f"{sender} {subject} {body}".lower()

    is_spam = any(kw in text for kw in SPAM_KEYWORDS) or ".xyz" in sender or "paypa1" in sender
    if is_spam:
        return {
            "email_id": email_id, "priority": "spam", "category": "spam_phishing",
            "route_to": "trash", "summary": f"Suspected spam/phishing: {subject[:80]}",
            "flag_review": False, "reasoning": "Spam signals detected.",
        }

    is_urgent = any(kw in text for kw in ["urgent", "hacked", "breach", "outage", "ransomware",
                                           "legal action", "72 hours", "deadline", "lawsuit"])
    is_legal  = any(kw in text for kw in ["attorney", "legal", "comply", "regulatory", "labor code",
                                           "sec", "fraud", "journalist", "acquisition"])
    is_billing= any(kw in text for kw in ["invoice", "payment", "charge", "billing", "overdue", "renewal"])
    is_sales  = any(kw in text for kw in ["enterprise", "pricing", "seats", "cto", "cfo", "contract", "rfp"])
    is_hr     = any(kw in text for kw in ["pip", "performance", "employee", "lunch", "hr", "wages", "misconduct"])
    is_tech   = any(kw in text for kw in ["api", "error", "outage", "server", "login", "account"])

    if is_legal:
        return {
            "email_id": email_id, "priority": "urgent" if is_urgent else "high",
            "category": "legal_compliance", "route_to": "legal" if not is_urgent else "management",
            "summary": f"Legal/compliance matter: {subject[:100]}",
            "flag_review": True, "reasoning": "Legal signals detected.",
        }
    if is_billing:
        return {
            "email_id": email_id, "priority": "high" if is_urgent else "medium",
            "category": "billing_inquiry", "route_to": "billing",
            "summary": f"Billing issue: {subject[:100]}",
            "flag_review": False, "reasoning": "Billing keywords.",
        }
    if is_sales:
        return {
            "email_id": email_id, "priority": "high",
            "category": "sales_lead", "route_to": "sales",
            "summary": f"Sales opportunity: {subject[:100]}",
            "flag_review": False, "reasoning": "Sales lead keywords.",
        }
    if is_hr:
        return {
            "email_id": email_id, "priority": "medium",
            "category": "internal_hr", "route_to": "hr",
            "summary": f"HR matter: {subject[:100]}",
            "flag_review": "misconduct" in text or "whistleblower" in text,
            "reasoning": "HR keywords.",
        }
    if is_tech or is_urgent:
        return {
            "email_id": email_id, "priority": "urgent" if is_urgent else "medium",
            "category": "technical_support", "route_to": "support_tier2" if is_urgent else "support_tier1",
            "summary": f"Technical issue: {subject[:100]}",
            "flag_review": is_urgent, "reasoning": "Technical/urgent signals.",
        }
    return {
        "email_id": email_id, "priority": "low",
        "category": "general_inquiry", "route_to": "support_tier1",
        "summary": f"General inquiry: {subject[:100]}",
        "flag_review": False, "reasoning": "No strong signals detected.",
    }


def run_task(task_id: str) -> Dict[str, Any]:
    """Run the baseline agent on a single task and return results."""
    print(f"\n{'='*60}")
    print(f"Task: {task_id.upper()}")
    print(f"{'='*60}")

    env = EmailTriageEnv(task_id=task_id, seed=SEED)
    obs = env.reset()

    print(f"  Emails to triage: {obs.total_emails}")
    episode_actions = []
    step_rewards    = []

    while not env.is_done:
        if obs.current_email is None:
            break

        email = obs.current_email
        h = email.header
        print(f"\n  [{h.email_id}] {h.subject[:60]}")

        if client:
            prompt = build_user_prompt(
                email_id=h.email_id, sender=h.sender, subject=h.subject,
                body=email.body, has_attachment=h.has_attachment, timestamp=h.timestamp,
            )
            action_dict = call_model(prompt)
        else:
            action_dict = rule_based_triage(h.email_id, h.sender, h.subject, email.body)

        if action_dict is None:
            print("    ERROR: Could not parse model response. Skipping.")
            continue

        # Ensure email_id is correct (model might hallucinate it)
        action_dict["email_id"] = h.email_id

        try:
            action = Action(**action_dict)
        except Exception as exc:
            print(f"    ERROR: Invalid action structure: {exc}")
            continue

        obs, reward, done, info = env.step(action)
        episode_actions.append(action_dict)
        step_rewards.append(reward.total)

        print(f"    → priority={action.priority.value} | category={action.category.value} "
              f"| route={action.route_to.value} | score={reward.total:.3f}")

    # Final episode summary
    summary = grade_episode(episode_actions)
    print(f"\n  ── Episode Summary ──────────────────────────")
    print(f"  Overall score:   {summary['label_score']:.4f}")
    print(f"  Emails triaged:  {summary['num_emails']}")
    print(f"  Min step score:  {summary['min_score']:.4f}")
    print(f"  Max step score:  {summary['max_score']:.4f}")

    return summary


def main():
    print("=" * 60)
    print("Email Triage OpenEnv — Baseline Inference")
    print(f"Model: {MODEL if client else 'rule-based-demo'}")
    print(f"Seed:  {SEED}")
    print("=" * 60)

    results = {}
    for task in TASKS:
        results[task] = run_task(task)

    print(f"\n{'='*60}")
    print("FINAL BASELINE SCORES")
    print(f"{'='*60}")
    for task, res in results.items():
        score = res["label_score"]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task:8s}  [{bar}]  {score:.4f}")

    overall = sum(r["label_score"] for r in results.values()) / len(results)
    print(f"\n  {'OVERALL':8s}  {overall:.4f}")
    print(f"{'='*60}\n")

    # Write results JSON
    output = {
        "model": MODEL if client else "rule-based-demo",
        "seed": SEED,
        "task_scores": {t: r["label_score"] for t, r in results.items()},
        "overall": overall,
        "details": results,
    }
    out_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {out_path}")
    return output


if __name__ == "__main__":
    main()
