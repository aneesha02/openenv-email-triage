"""
Typed Pydantic models for the Email Triage OpenEnv environment.
v2: adds sequential session state — escalation budget, SLA timers,
team queue capacities — so agent decisions have lasting cross-step effects.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Priority(str, Enum):
    URGENT = "urgent"
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"
    SPAM   = "spam"


class Category(str, Enum):
    CUSTOMER_COMPLAINT = "customer_complaint"
    BILLING_INQUIRY    = "billing_inquiry"
    TECHNICAL_SUPPORT  = "technical_support"
    SALES_LEAD         = "sales_lead"
    INTERNAL_HR        = "internal_hr"
    LEGAL_COMPLIANCE   = "legal_compliance"
    SPAM_PHISHING      = "spam_phishing"
    GENERAL_INQUIRY    = "general_inquiry"


class RouteTo(str, Enum):
    SUPPORT_TIER1 = "support_tier1"
    SUPPORT_TIER2 = "support_tier2"
    BILLING       = "billing"
    SALES         = "sales"
    LEGAL         = "legal"
    HR            = "hr"
    MANAGEMENT    = "management"
    TRASH         = "trash"
    ARCHIVE       = "archive"


# ── Sequential session constants ──────────────────────────────────────────────

TEAM_CAPACITY: Dict[str, int] = {
    "support_tier1": 3,
    "support_tier2": 2,
    "billing":       3,
    "sales":         3,
    "legal":         2,
    "hr":            2,
    "management":    2,
    "trash":         99,
    "archive":       99,
}

# Steps before an unhandled email breaches SLA
SLA_STEPS: Dict[str, int] = {
    "urgent": 2,
    "high":   4,
    "medium": 8,
    "low":    99,
    "spam":   99,
}
TASK_ESCALATION_BUDGET: Dict[str, int] = {
    "easy":   3,
    "medium": 4,
    "hard":   5,
}


class TeamQueueState(BaseModel):
    support_tier1: int = TEAM_CAPACITY["support_tier1"]
    support_tier2: int = TEAM_CAPACITY["support_tier2"]
    billing:       int = TEAM_CAPACITY["billing"]
    sales:         int = TEAM_CAPACITY["sales"]
    legal:         int = TEAM_CAPACITY["legal"]
    hr:            int = TEAM_CAPACITY["hr"]
    management:    int = TEAM_CAPACITY["management"]
    trash:         int = TEAM_CAPACITY["trash"]
    archive:       int = TEAM_CAPACITY["archive"]

    def remaining(self, route: str) -> int:
        return getattr(self, route, 0)

    def consume(self, route: str) -> bool:
        cap = getattr(self, route, 0)
        if cap <= 0:
            return False
        setattr(self, route, cap - 1)
        return True


class SlaStatus(BaseModel):
    email_id:        str
    true_priority:   str
    arrived_at_step: int
    deadline_step:   int
    breached:        bool = False


class SessionConstraints(BaseModel):
    """Shared state that persists across every step — makes this a true sequential problem."""
    escalation_budget: int = 3
    escalations_used:  int = 0
    team_queues:       TeamQueueState  = Field(default_factory=TeamQueueState)
    sla_tracker:       List[SlaStatus] = Field(default_factory=list)
    sla_breaches:      int  = 0
    queue_overflows:   int  = 0
    cascade_triggered: bool = False


# ── Observation ───────────────────────────────────────────────────────────────

class EmailHeader(BaseModel):
    email_id:       str
    sender:         str
    subject:        str
    timestamp:      str
    thread_id:      Optional[str] = None
    has_attachment: bool = False


class EmailMessage(BaseModel):
    header:   EmailHeader
    body:     str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    inbox:         List[EmailMessage]        = Field(default_factory=list)
    processed:     List[str]                = Field(default_factory=list)
    current_email: Optional[EmailMessage]   = None
    step_number:   int                      = 0
    total_emails:  int                      = 0
    remaining:     int                      = 0
    # Sequential state (agent-visible)
    escalation_budget_remaining: int              = 3
    team_queue_remaining:        Dict[str, int]   = Field(default_factory=dict)
    active_sla_warnings:         List[Dict]       = Field(default_factory=list)
    sla_breaches_so_far:         int              = 0
    cascade_active:              bool             = False
    session_info:                Dict[str, Any]   = Field(default_factory=dict)


# ── Action ────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    email_id:    str
    priority:    Priority
    category:    Category
    route_to:    RouteTo
    summary:     str  = Field(..., max_length=280)
    flag_review: bool = Field(False)
    reasoning:   str  = Field("")


# ── Reward ────────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    priority_score:   float = 0.0
    category_score:   float = 0.0
    routing_score:    float = 0.0
    summary_score:    float = 0.0
    escalation_score: float = 0.0
    sla_penalty:      float = 0.0
    queue_penalty:    float = 0.0
    budget_penalty:   float = 0.0
    cascade_penalty:  float = 0.0
    base_penalty:     float = 0.0


class Reward(BaseModel):
    total:     float           = 0.0
    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    done:      bool            = False
    info:      Dict[str, Any]  = Field(default_factory=dict)


# ── State snapshot ────────────────────────────────────────────────────────────

class EnvironmentState(BaseModel):
    task_id:           str
    step:              int
    done:              bool
    observation:       Observation
    cumulative_reward: float
    actions_taken:     List[Dict[str, Any]] = Field(default_factory=list)
    grader_scores:     Dict[str, float]     = Field(default_factory=dict)
    constraints:       Dict[str, Any]       = Field(default_factory=dict)
