"""
Agent graders for each task difficulty level.
All graders return a float score in [0.0, 1.0].

score_action() grades pure label correctness (priority/category/route/summary/escalation).
Sequential penalties (SLA, budget, queue) are applied by environment.py, not here.
grade_episode() aggregates label scores across a full episode for reporting.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from models import Action, Priority, RouteTo, RewardBreakdown, Reward
from dataset import ALL_EMAILS_BY_ID

PRIORITY_WEIGHT   = 0.35
CATEGORY_WEIGHT   = 0.25
ROUTING_WEIGHT    = 0.25
SUMMARY_WEIGHT    = 0.10
ESCALATION_WEIGHT = 0.05

PRIORITY_ADJACENCY: Dict[str, Dict[str, float]] = {
    "urgent": {"urgent":1.0,"high":0.5,"medium":0.1,"low":0.0,"spam":0.0},
    "high":   {"urgent":0.5,"high":1.0,"medium":0.5,"low":0.1,"spam":0.0},
    "medium": {"urgent":0.1,"high":0.5,"medium":1.0,"low":0.5,"spam":0.0},
    "low":    {"urgent":0.0,"high":0.1,"medium":0.5,"low":1.0,"spam":0.2},
    "spam":   {"urgent":0.0,"high":0.0,"medium":0.0,"low":0.2,"spam":1.0},
}

RELATED_CATEGORIES: Dict[Tuple[str,str], float] = {
    ("customer_complaint","billing_inquiry"):0.4,
    ("billing_inquiry","customer_complaint"):0.4,
    ("legal_compliance","customer_complaint"):0.2,
    ("customer_complaint","legal_compliance"):0.2,
    ("technical_support","customer_complaint"):0.3,
    ("customer_complaint","technical_support"):0.3,
    ("internal_hr","legal_compliance"):0.2,
    ("legal_compliance","internal_hr"):0.2,
    ("general_inquiry","billing_inquiry"):0.3,
}


def _category_score(predicted: str, actual: str) -> float:
    if predicted == actual:
        return 1.0
    return RELATED_CATEGORIES.get((predicted, actual), 0.0)


def _routing_score(predicted: str, actual: str) -> float:
    if predicted == actual:
        return 1.0
    acceptable = {
        "support_tier2": ["support_tier1"],
        "support_tier1": ["support_tier2"],
        "management":    ["legal"],
        "legal":         ["management"],
        "hr":            ["management"],
        "trash":         ["archive"],
        "archive":       ["trash"],
    }
    if predicted in acceptable.get(actual, []):
        return 0.4
    return 0.0


def _summary_score(summary: str, body: str, subject: str) -> float:
    if not summary or len(summary) < 10:
        return 0.0
    score = 0.0
    if 30 <= len(summary) <= 280:
        score += 0.4
    if summary.strip().lower() != subject.strip().lower():
        score += 0.2
    body_words   = set(re.findall(r'\b\w{5,}\b', body.lower()))
    summ_words   = set(re.findall(r'\b\w{5,}\b', summary.lower()))
    overlap      = body_words & summ_words
    score += 0.3 if len(overlap) >= 3 else (0.1 if overlap else 0.0)
    if not summary.strip().startswith("I "):
        score += 0.1
    return min(score, 1.0)


def score_action(action: Action) -> Tuple[Reward, Dict[str, Any]]:
    """
    Grade label correctness for one action. Returns (Reward, detail).
    Sequential penalties (SLA, budget, queue) are NOT included here —
    they are added by EmailTriageEnv.step().
    """
    email_data = ALL_EMAILS_BY_ID.get(action.email_id)
    if email_data is None:
        bd = RewardBreakdown(base_penalty=-0.5)
        return Reward(total=-0.5, breakdown=bd,
                      info={"error": f"Unknown email_id: {action.email_id}"}), {"error":"unknown_id"}

    gt      = email_data["ground_truth"]
    email   = email_data["email"]
    subject = email["header"]["subject"]
    body    = email["body"]

    pri  = PRIORITY_ADJACENCY.get(gt["priority"], {}).get(action.priority.value, 0.0)
    cat  = _category_score(action.category.value, gt["category"])
    rte  = _routing_score(action.route_to.value, gt["route_to"])
    summ = _summary_score(action.summary, body, subject)

    esc_needed = gt.get("requires_escalation", False)
    if action.flag_review == esc_needed:
        esc = 1.0
    elif action.flag_review and not esc_needed:
        esc = 0.5
    else:
        esc = 0.0

    base_penalty = 0.0
    if gt["category"] == "spam_phishing" and action.route_to not in (RouteTo.TRASH, RouteTo.ARCHIVE):
        base_penalty -= 0.2
    if gt["priority"] == "urgent" and action.priority == Priority.SPAM:
        base_penalty -= 0.3

    bd = RewardBreakdown(
        priority_score   = pri,
        category_score   = cat,
        routing_score    = rte,
        summary_score    = summ,
        escalation_score = esc,
        base_penalty     = base_penalty,
    )
    total = max(0.0, min(1.0,
        pri*PRIORITY_WEIGHT + cat*CATEGORY_WEIGHT + rte*ROUTING_WEIGHT
        + summ*SUMMARY_WEIGHT + esc*ESCALATION_WEIGHT + base_penalty
    ))

    detail = {
        "email_id":      action.email_id,
        "gt_priority":   gt["priority"],
        "gt_category":   gt["category"],
        "gt_route":      gt["route_to"],
        "gt_escalate":   esc_needed,
        "pred_priority": action.priority.value,
        "pred_category": action.category.value,
        "pred_route":    action.route_to.value,
        "pred_escalate": action.flag_review,
        "scores": {
            "priority":   round(pri,  3),
            "category":   round(cat,  3),
            "routing":    round(rte,  3),
            "summary":    round(summ, 3),
            "escalation": round(esc,  3),
            "base_penalty": round(base_penalty, 3),
        },
        "label_total": round(total, 3),
    }
    return Reward(total=total, breakdown=bd, info=detail), detail


def grade_episode(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate label-correctness scores across a full episode.
    Note: this does NOT include sequential penalties (SLA/budget/queue) —
    use the per-step rewards from env.step() for the full picture.
    """
    per_email: List[Dict] = []
    totals: List[float]   = []
    for a_dict in actions:
        try:
            action = Action(**a_dict)
            reward, detail = score_action(action)
            per_email.append(detail)
            totals.append(reward.total)
        except Exception as exc:
            per_email.append({"error": str(exc), "total": 0.0})
            totals.append(0.0)

    overall = sum(totals) / len(totals) if totals else 0.0
    return {
        "label_score":      round(overall, 4),   # label correctness only
        "num_emails":       len(totals),
        "per_email_scores": per_email,
        "min_score":        round(min(totals), 4) if totals else 0.0,
        "max_score":        round(max(totals), 4) if totals else 0.0,
    }
