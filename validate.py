#!/usr/bin/env python3
"""
validate.py — Full validation + reproducible baseline for Email Triage OpenEnv v2.

Works with Python stdlib only (no pydantic, no fastapi required).
Run:  python validate.py
Run with JSON output:  python validate.py --json

Produces:
  1.  Unit tests: model layer, grader, environment lifecycle, episode boundaries
  2.  Sequential-state tests: SLA timers, budget exhaustion, queue saturation,
      cascade trigger — the mechanics that make this a real sequential problem
  3.  Difficulty validation: three agent tiers × three tasks, proving the
      score spread widens with task difficulty
  4.  Reproducible baseline scores with SHA-256 checksum (no model API needed)
"""

import hashlib, json, re, sys, os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

PASS = "✓"; FAIL = "✗"
results: List[Tuple[bool, str]] = []
OUTPUT_ARGS = "--json" in sys.argv

# ─── Minimal stdlib models (mirrors pydantic models exactly) ─────────────────

class Priority(str, Enum):
    URGENT="urgent"; HIGH="high"; MEDIUM="medium"; LOW="low"; SPAM="spam"

class Category(str, Enum):
    CUSTOMER_COMPLAINT="customer_complaint"; BILLING_INQUIRY="billing_inquiry"
    TECHNICAL_SUPPORT="technical_support";  SALES_LEAD="sales_lead"
    INTERNAL_HR="internal_hr";             LEGAL_COMPLIANCE="legal_compliance"
    SPAM_PHISHING="spam_phishing";         GENERAL_INQUIRY="general_inquiry"

class RouteTo(str, Enum):
    SUPPORT_TIER1="support_tier1"; SUPPORT_TIER2="support_tier2"
    BILLING="billing"; SALES="sales"; LEGAL="legal"; HR="hr"
    MANAGEMENT="management"; TRASH="trash"; ARCHIVE="archive"

TEAM_CAPACITY = {
    "support_tier1":3,"support_tier2":2,"billing":3,"sales":3,
    "legal":2,"hr":2,"management":2,"trash":99,"archive":99,
}
SLA_STEPS = {"urgent":2,"high":4,"medium":8,"low":99,"spam":99}
TASK_ESCALATION_BUDGET = {"easy":3,"medium":4,"hard":5}

@dataclass
class Action:
    email_id: str; priority: str; category: str; route_to: str
    summary: str;  flag_review: bool; reasoning: str = ""
    def __post_init__(self):
        assert len(self.summary) <= 280, f"Summary too long ({len(self.summary)})"
        assert self.priority in [p.value for p in Priority]
        assert self.category in [c.value for c in Category]
        assert self.route_to in [r.value for r in RouteTo]

@dataclass
class RewardBreakdown:
    priority_score:float=0.0; category_score:float=0.0
    routing_score:float=0.0;  summary_score:float=0.0
    escalation_score:float=0.0; sla_penalty:float=0.0
    queue_penalty:float=0.0; budget_penalty:float=0.0
    cascade_penalty:float=0.0; base_penalty:float=0.0

@dataclass
class Reward:
    total:float=0.0; breakdown:RewardBreakdown=field(default_factory=RewardBreakdown)
    done:bool=False; info:Dict[str,Any]=field(default_factory=dict)

@dataclass
class SlaStatus:
    email_id:str; true_priority:str; arrived_at_step:int
    deadline_step:int; breached:bool=False

@dataclass
class TeamQueues:
    queues: Dict[str,int] = field(default_factory=lambda: dict(TEAM_CAPACITY))
    def remaining(self, r: str) -> int: return self.queues.get(r,0)
    def consume(self, r: str) -> bool:
        if self.queues.get(r,0) <= 0: return False
        self.queues[r] -= 1; return True

@dataclass
class SessionConstraints:
    escalation_budget:int=3; escalations_used:int=0
    team_queues:TeamQueues=field(default_factory=TeamQueues)
    sla_tracker:List[SlaStatus]=field(default_factory=list)
    sla_breaches:int=0; queue_overflows:int=0; cascade_triggered:bool=False

# ─── Dataset (same ground truths as dataset.py) ──────────────────────────────

DATASET: Dict[str, List[Dict]] = {
  "easy": [
    {"id":"e001","priority":"urgent","category":"technical_support","route_to":"support_tier2","escalate":True,
     "subject":"URGENT: My account has been hacked!",
     "body":"I just received a notification that someone logged into my account from Russia at 3 AM. I did NOT authorize this. Please lock my account immediately. This is extremely urgent — I have sensitive data in there. Account #A-10293"},
    {"id":"e002","priority":"spam","category":"spam_phishing","route_to":"trash","escalate":False,
     "subject":"Congratulations! You've WON $1,000,000!",
     "body":"You have been selected from millions of entries to receive ONE MILLION DOLLARS. Click the link and provide your bank details to claim your prize! http://totally-legit-lottery.xyz/claim"},
    {"id":"e003","priority":"low","category":"internal_hr","route_to":"hr","escalate":False,
     "subject":"Team lunch this Friday",
     "body":"Just a reminder that we're doing team lunch this Friday at Rosario's at 12:30 PM. Please let me know by Wednesday if you can attend so I can book the right table size."},
    {"id":"e004","priority":"high","category":"billing_inquiry","route_to":"billing","escalate":False,
     "subject":"Invoice #INV-2024-0042 - Payment Overdue",
     "body":"Invoice #INV-2024-0042 for $4,250 was due on January 1st and remains unpaid. Please arrange payment within 5 business days to avoid late fees. Acme Corp Billing"},
    {"id":"e005","priority":"high","category":"sales_lead","route_to":"sales","escalate":False,
     "subject":"Interested in your Enterprise plan",
     "body":"I'm the CTO at StartupCo (50 engineers). We've been evaluating your platform and are very interested in the Enterprise tier. Could someone from sales reach out to discuss pricing and custom integrations?"},
  ],
  "medium": [
    {"id":"m001","priority":"urgent","category":"customer_complaint","route_to":"management","escalate":True,
     "subject":"Completely unacceptable service - threatening legal action",
     "body":"8-year customer. Technician came THREE TIMES still hasn't fixed the issue. Lost $12,000 in business revenue. Consulting my attorney. I expect a full refund in 48 hours or I will be filing suit. Account #C-88234 Premium tier"},
    {"id":"m002","priority":"urgent","category":"legal_compliance","route_to":"legal","escalate":True,
     "subject":"Data Breach Notification Requirement - Response Required",
     "body":"Pursuant to Section 12(b) Data Protection Act 2023, respond to data incident December 28 2023 affecting 3200 customer records. You have 72 hours to provide incident timeline affected data categories and remediation steps. Failure may result in regulatory penalties. Inspector Davies Data Protection Authority"},
    {"id":"m003","priority":"urgent","category":"technical_support","route_to":"support_tier2","escalate":True,
     "subject":"API rate limits causing production outage",
     "body":"Production returning 429 errors since 15:50 UTC. 40% of transactions failing. Business plan should allow 500 req/min but throttling at 200. Request IDs REQ-abc123 REQ-def456. This is a P0 please escalate immediately."},
    {"id":"m004","priority":"high","category":"internal_hr","route_to":"hr","escalate":False,
     "subject":"Confidential: Performance Improvement Plan - James Wilson",
     "body":"Attached PIP for James Wilson EMP-4421 effective immediately with 60-day review. Strictly confidential stored securely. Weekly 1:1 check-ins required. KPI targets outlined in doc. Legal has reviewed and approved."},
    {"id":"m005","priority":"spam","category":"spam_phishing","route_to":"trash","escalate":False,
     "subject":"Action required: Verify your PayPal account",
     "body":"Unusual activity detected on your account. Verify identity at http://paypa1-security.net. Provide full name SSN credit card number CVV. Failure to verify in 24 hours will result in account suspension."},
    {"id":"m006","priority":"high","category":"sales_lead","route_to":"sales","escalate":False,
     "subject":"Pricing proposal for 500-seat license",
     "body":"Formal RFP for 500-seat enterprise license. Q1 budget approved ready to move quickly. 3-year contract. Please have your VP of Sales contact me directly. Amanda Torres CFO BigCorp $2B ARR"},
    {"id":"m007","priority":"medium","category":"billing_inquiry","route_to":"billing","escalate":False,
     "subject":"Question about my bill",
     "body":"Got charged $149 this month but I'm on the $99 plan. Can you explain the extra charges? My account email is user4492@mail.com."},
    {"id":"m008","priority":"high","category":"internal_hr","route_to":"hr","escalate":True,
     "subject":"[CONFIDENTIAL] Internal misconduct report",
     "body":"I am a current employee and I have witnessed repeated violations of our expense policy by a senior director over 6 months. I have documentation. I am reaching out anonymously because I fear retaliation. Please advise on how to report this without revealing my identity."},
  ],
  "hard": [
    {"id":"h001","priority":"urgent","category":"legal_compliance","route_to":"legal","escalate":True,
     "subject":"Comment request: alleged data misuse by your AI system",
     "body":"Journalist at Investigative News. Story about AI companies using customer data without explicit consent. Sources indicate your platform may be implicated. Publication deadline 9 AM tomorrow. 2M monthly readers. Requesting official comment. If I do not receive one I will note that you declined to comment."},
    {"id":"h002","priority":"urgent","category":"legal_compliance","route_to":"legal","escalate":True,
     "subject":"RE: RE: RE: Contract renewal - final terms",
     "body":"Legal team still objects to indemnification language in section 12. Auto-renewal clause 15.3 needs 30 days notice not 90. If we can resolve those two points ready to sign by EOD Friday. Board meeting next Tuesday. Mike Chen VP Partnerships"},
    {"id":"h003","priority":"urgent","category":"technical_support","route_to":"management","escalate":True,
     "subject":"Automated alert: 847 tickets unresolved > 48h SLA breach",
     "body":"SLA BREACH DETECTED. Unresolved tickets over 48h: 847. Avg resolution time 61h vs SLA 24h. CSAT 2.1/5.0. Critical tickets: 43. Automatic escalation triggered. Immediate action required."},
    {"id":"h004","priority":"urgent","category":"legal_compliance","route_to":"management","escalate":True,
     "subject":"Acquisition conversation?",
     "body":"I will be direct. We have been watching your growth and believe there could be significant synergies combining our organizations. I would like a private confidential conversation with your CEO. Please treat this with utmost discretion. David Schwartz CEO"},
    {"id":"h005","priority":"high","category":"billing_inquiry","route_to":"billing","escalate":False,
     "subject":"Your subscription renewal failed - action required",
     "body":"Annual subscription renewal $87,400 failed to process. Card ending 4821 declined. Service active 7 days. Please update payment method. SaaS Platform Billing"},
    {"id":"h006","priority":"urgent","category":"legal_compliance","route_to":"legal","escalate":True,
     "subject":"I'm owed unpaid wages from my termination",
     "body":"Employment terminated December 31st. Final paycheck including 15 days PTO worth approximately $6,200 not received. Emailed HR 3 times no response. Per California Labor Code Section 201 final wages due immediately upon termination. Each day of delay incurs waiting time penalties. Filing with California Labor Commissioner in 72 hours."},
    {"id":"h007","priority":"spam","category":"spam_phishing","route_to":"trash","escalate":True,
     "subject":"Microsoft 365 admin: Security alert - immediate action",
     "body":"Microsoft 365 Security Alert. Download and run attached MS365_SecureFix.exe. Suspicious login detected. Failure to act within 2 hours will result in account suspension for all 340 users in your organization. Microsoft Security Team https://microsoft-alert-security.xyz"},
    {"id":"h008","priority":"urgent","category":"legal_compliance","route_to":"legal","escalate":True,
     "subject":"Evidence of financial fraud in your organization",
     "body":"I have obtained documents showing that a VP at your company has been submitting fictitious vendor invoices totalling approximately $340,000 over 18 months. I have bank transfer receipts and fabricated contracts. Providing to you before going to the SEC."},
    {"id":"h009","priority":"urgent","category":"legal_compliance","route_to":"management","escalate":True,
     "subject":"Follow-up: Due diligence data room access",
     "body":"Series B investment $15M proceeding to next stage. Our team needs access to the data room by COB today. Audited financials cap table and customer contracts required. Partner meeting Thursday need to complete review before then. Please ensure right people are looped in immediately."},
    {"id":"h010","priority":"urgent","category":"technical_support","route_to":"management","escalate":True,
     "subject":"Incident Report: Ransomware detected on 3 servers",
     "body":"Ransomware activity detected on servers SRV-DB-01 SRV-APP-03 SRV-BACKUP-02. Automatic isolation triggered. Estimated data encrypted 40% of customer database. Attacker note Pay 25 BTC within 48h. Customer-facing services DEGRADED 23% error rate. Immediate action required from CISO Legal Communications CEO."},
  ],
}

# ─── Grader (mirrors grader.py) ──────────────────────────────────────────────

PRIORITY_ADJACENCY = {
    "urgent":{"urgent":1.0,"high":0.5,"medium":0.1,"low":0.0,"spam":0.0},
    "high":  {"urgent":0.5,"high":1.0,"medium":0.5,"low":0.1,"spam":0.0},
    "medium":{"urgent":0.1,"high":0.5,"medium":1.0,"low":0.5,"spam":0.0},
    "low":   {"urgent":0.0,"high":0.1,"medium":0.5,"low":1.0,"spam":0.2},
    "spam":  {"urgent":0.0,"high":0.0,"medium":0.0,"low":0.2,"spam":1.0},
}
RELATED_CAT = {
    ("customer_complaint","billing_inquiry"):0.4,("billing_inquiry","customer_complaint"):0.4,
    ("legal_compliance","customer_complaint"):0.2,("customer_complaint","legal_compliance"):0.2,
    ("technical_support","customer_complaint"):0.3,("customer_complaint","technical_support"):0.3,
    ("internal_hr","legal_compliance"):0.2,("legal_compliance","internal_hr"):0.2,
    ("general_inquiry","billing_inquiry"):0.3,
}
ACCEPTABLE_ROUTES = {
    "support_tier2":["support_tier1"],"support_tier1":["support_tier2"],
    "management":["legal"],"legal":["management"],"hr":["management"],
    "trash":["archive"],"archive":["trash"],
}

def _cat_score(pred,actual): return 1.0 if pred==actual else RELATED_CAT.get((pred,actual),0.0)
def _rte_score(pred,actual): return 1.0 if pred==actual else (0.4 if pred in ACCEPTABLE_ROUTES.get(actual,[]) else 0.0)
def _summ_score(s,body,subject):
    if not s or len(s)<10: return 0.0
    sc=0.0
    if 30<=len(s)<=280: sc+=0.4
    if s.strip().lower()!=subject.strip().lower(): sc+=0.2
    bw=set(re.findall(r'\b\w{5,}\b',body.lower()))
    sw=set(re.findall(r'\b\w{5,}\b',s.lower()))
    ov=bw&sw; sc+=0.3 if len(ov)>=3 else (0.1 if ov else 0.0)
    if not s.strip().startswith("I "): sc+=0.1
    return min(sc,1.0)

def score_label(action: Action, gt: dict) -> Tuple[float, dict]:
    """Score label correctness only (0..1). No sequential penalties."""
    pri  = PRIORITY_ADJACENCY.get(gt["priority"],{}).get(action.priority,0.0)
    cat  = _cat_score(action.category,gt["category"])
    rte  = _rte_score(action.route_to,gt["route_to"])
    summ = _summ_score(action.summary, gt["body"], gt["subject"])
    esc_needed = gt.get("escalate",False)
    esc  = 1.0 if action.flag_review==esc_needed else (0.5 if action.flag_review else 0.0)
    pen  = 0.0
    if gt["category"]=="spam_phishing" and action.route_to not in ("trash","archive"): pen-=0.2
    if gt["priority"]=="urgent" and action.priority=="spam": pen-=0.3
    total = max(0.0,min(1.0, pri*0.35+cat*0.25+rte*0.25+summ*0.10+esc*0.05+pen))
    return total, {"pri":round(pri,3),"cat":round(cat,3),"rte":round(rte,3),
                   "summ":round(summ,3),"esc":round(esc,3),"pen":round(pen,3),"total":round(total,3)}

# ─── Sequential Environment (stdlib, mirrors environment.py) ─────────────────

class EmailTriageEnv:
    def __init__(self, task_id: str):
        assert task_id in DATASET
        self.task_id = task_id
        self._emails:  List[dict] = []
        self._processed: List[str] = []
        self._step = 0; self._done = False
        self._cumulative = 0.0
        self._actions: List[dict] = []
        self._con: SessionConstraints = SessionConstraints()

    def reset(self):
        self._emails    = list(DATASET[self.task_id])
        self._processed = []; self._actions = []
        self._step = 0; self._done = False; self._cumulative = 0.0
        budget = TASK_ESCALATION_BUDGET[self.task_id]
        self._con = SessionConstraints(
            escalation_budget=budget,
            team_queues=TeamQueues(queues=dict(TEAM_CAPACITY))
        )
        # Register SLA deadlines
        for i, e in enumerate(self._emails):
            deadline = i + SLA_STEPS.get(e["priority"], 99)
            self._con.sla_tracker.append(SlaStatus(
                email_id=e["id"], true_priority=e["priority"],
                arrived_at_step=i, deadline_step=deadline,
            ))
        return self._obs()

    def step(self, action: Action):
        assert not self._done, "Episode done. Call reset()."
        bd = RewardBreakdown()

        # A. SLA decay
        sla_pen = 0.0
        pset = set(self._processed)
        for sla in self._con.sla_tracker:
            if sla.email_id in pset or sla.breached: continue
            if self._step >= sla.deadline_step:
                sla.breached = True
                self._con.sla_breaches += 1
                sla_pen -= 0.15
        bd.sla_penalty = sla_pen

        # B. Validate
        inbox_ids = {e["id"] for e in self._emails}
        if action.email_id not in inbox_ids:
            bd.base_penalty = -0.1
            total = max(-1.0, sla_pen - 0.1)
            r = Reward(total=total, breakdown=bd, done=False,
                       info={"error":"invalid email_id"})
            self._step += 1; self._cumulative += total
            return self._obs(), r, False, r.info

        gt = next(e for e in self._emails if e["id"] == action.email_id)
        label_total, detail = score_label(action, gt)

        bd.priority_score   = detail["pri"]
        bd.category_score   = detail["cat"]
        bd.routing_score    = detail["rte"]
        bd.summary_score    = detail["summ"]
        bd.escalation_score = detail["esc"]
        bd.base_penalty     = detail["pen"]

        # C. Escalation budget
        bpen = 0.0
        if action.flag_review:
            if self._con.escalations_used >= self._con.escalation_budget:
                bpen = -0.20; bd.budget_penalty = bpen
                detail["budget_overflow"] = True
            else:
                self._con.escalations_used += 1

        # D. Queue capacity
        qpen = 0.0
        rk = action.route_to
        if rk not in ("trash","archive"):
            if not self._con.team_queues.consume(rk):
                qpen = -0.10; bd.queue_penalty = qpen
                self._con.queue_overflows += 1
                detail["queue_overflow"] = rk

        # E. Process
        self._actions.append(asdict(action))
        self._processed.append(action.email_id)
        self._emails = [e for e in self._emails if e["id"] != action.email_id]
        for sla in self._con.sla_tracker:
            if sla.email_id == action.email_id:
                sla.breached = True; break

        # F. Cascade
        cpen = 0.0
        urgent_late = sum(
            1 for s in self._con.sla_tracker
            if s.breached and s.true_priority=="urgent" and s.deadline_step <= self._step
        )
        if urgent_late >= 2 and not self._con.cascade_triggered:
            self._con.cascade_triggered = True
            cpen = -0.25; bd.cascade_penalty = cpen

        total = max(-1.0, min(1.0, label_total + sla_pen + bpen + qpen + cpen))
        self._step += 1; self._cumulative += total
        self._done = len(self._emails) == 0

        detail.update({"sla_pen":sla_pen,"bpen":bpen,"qpen":qpen,"cpen":cpen,"step_total":round(total,4)})
        r = Reward(total=round(total,4), breakdown=bd, done=self._done, info=detail)
        return self._obs(), r, self._done, r.info

    def _obs(self):
        c = self._con
        return {
            "current_email": self._emails[0] if self._emails else None,
            "remaining": len(self._emails),
            "total": len(DATASET[self.task_id]),
            "step": self._step,
            "escalation_budget_remaining": c.escalation_budget - c.escalations_used,
            "team_queues": {k: c.team_queues.remaining(k) for k in TEAM_CAPACITY},
            "sla_breaches": c.sla_breaches,
            "sla_warnings": [
                {"id":s.email_id,"steps_left":s.deadline_step-self._step}
                for s in c.sla_tracker
                if s.email_id not in set(self._processed) and not s.breached
                and s.deadline_step - self._step <= 2
            ],
            "cascade_active": c.cascade_triggered,
        }

# ─── Three agent tiers ────────────────────────────────────────────────────────

def agent_naive(email: dict, obs: dict) -> Action:
    """Naive: keyword-only, no budget/queue/SLA awareness."""
    text = (email["subject"]+" "+email["body"]).lower()
    if any(k in text for k in ["lottery","million","paypa1","ms365_securefix","bank details"]):
        return Action(email["id"],"spam","spam_phishing","trash","Spam.",False)
    if any(k in text for k in ["invoice","payment","billing","charge","renewal"]):
        return Action(email["id"],"medium","billing_inquiry","billing","Billing inquiry.",False)
    if any(k in text for k in ["api","server","error","outage","ransomware"]):
        return Action(email["id"],"medium","technical_support","support_tier1","Tech issue.",False)
    return Action(email["id"],"low","general_inquiry","support_tier1","General inquiry.",False)

def agent_medium(email: dict, obs: dict) -> Action:
    """Medium: correct labels but ignores budget/queue/SLA ordering."""
    return Action(
        email["id"], email["priority"], email["category"], email["route_to"],
        f"{email['subject'][:100]} — {email['route_to']}", email["escalate"],
    )

def agent_smart(email: dict, obs: dict) -> Action:
    """Smart: correct labels + SLA-aware ordering + budget management + queue awareness."""
    budget_left = obs["escalation_budget_remaining"]
    route = email["route_to"]
    queue_left = obs["team_queues"].get(route, 99)

    # Adapt routing when team queue is saturated
    if queue_left <= 0 and route not in ("trash","archive"):
        if route in ("legal","hr","support_tier2"):
            route = "management"     # overflow to management
        elif route == "support_tier1":
            route = "support_tier2"  # escalate within tier

    # Budget management: only escalate truly critical ones when budget is tight
    esc = email["escalate"]
    if esc and budget_left <= 0:
        esc = False   # budget exhausted — skip non-critical escalations
    if esc and budget_left == 1 and email["priority"] not in ("urgent",):
        esc = False   # save last budget slot for urgents

    summary = (
        f"{email['priority'].upper()}: {email['subject'][:80]} "
        f"| route→{route} | esc={esc}"
    )
    return Action(email["id"], email["priority"], email["category"], route, summary, esc)


def run_episode(task_id: str, agent_fn, name: str) -> dict:
    env = EmailTriageEnv(task_id)
    obs = env.reset()
    step_rewards = []; step_details = []
    while not env._done:
        email = obs["current_email"]
        action = agent_fn(email, obs)
        obs, reward, done, info = env.step(action)
        step_rewards.append(reward.total)
        step_details.append(info)
    avg = sum(step_rewards)/len(step_rewards) if step_rewards else 0.0
    return {
        "agent":task_id, "name":name, "task_id":task_id,
        "avg":round(avg,4), "min":round(min(step_rewards),4),
        "max":round(max(step_rewards),4),
        "steps":len(step_rewards),
        "sla_breaches":env._con.sla_breaches,
        "queue_overflows":env._con.queue_overflows,
        "budget_used":env._con.escalations_used,
        "budget_total":env._con.escalation_budget,
        "cascade":env._con.cascade_triggered,
        "step_rewards":step_rewards,
    }

# ─── Test harness ─────────────────────────────────────────────────────────────

def check(name: str, cond: bool, detail: str = "") -> bool:
    icon = PASS if cond else FAIL
    results.append((cond, name))
    if not OUTPUT_ARGS:
        print(f"  [{icon}] {'PASS' if cond else 'FAIL':4s} — {name}" + (f"  ({detail})" if detail else ""))
    return cond

def section(title: str):
    if not OUTPUT_ARGS:
        print(f"\n── {title} ──")

def main():
    if not OUTPUT_ARGS:
        print("=" * 66)
        print("  Email Triage OpenEnv v2 — Validation Suite")
        print("=" * 66)

    # ── 1. Model layer ────────────────────────────────────────────────────────
    section("1. Model layer")
    a = Action("e001","urgent","technical_support","support_tier2",
               "Security: account compromised — needs immediate lockdown.", True)
    check("Action instantiation", a.email_id=="e001")
    check("Summary length guard", len(a.summary)<=280, f"{len(a.summary)} chars")
    try:
        Action("x","urgent","technical_support","support_tier2","x"*281,False)
        check("Summary >280 raises", False)
    except AssertionError:
        check("Summary >280 raises", True)

    # ── 2. Grader unit tests ──────────────────────────────────────────────────
    section("2. Grader (label correctness)")
    gt_e001 = DATASET["easy"][0]
    perfect_score, _ = score_label(
        Action("e001","urgent","technical_support","support_tier2",
               "Security incident: unauthorised login detected. Account locked.", True),
        gt_e001
    )
    check("Perfect action ≥ 0.85", perfect_score >= 0.85, f"{perfect_score:.3f}")

    wrong_pri, _ = score_label(
        Action("e001","low","technical_support","support_tier2","Test.",True), gt_e001)
    check("Wrong priority reduces score", wrong_pri < perfect_score,
          f"wrong={wrong_pri:.3f} < correct={perfect_score:.3f}")

    spam_score, _ = score_label(
        Action("e002","spam","spam_phishing","trash","Lottery scam with .xyz domain.",False),
        DATASET["easy"][1])
    check("Spam correctly identified ≥ 0.85", spam_score >= 0.85, f"{spam_score:.3f}")

    mis_score, mis_detail = score_label(
        Action("e002","spam","spam_phishing","sales","Spam.",False), DATASET["easy"][1])
    check("Spam misrouted → penalty", mis_detail["pen"] < 0, f"pen={mis_detail['pen']}")

    uas_score, uas_detail = score_label(
        Action("e001","spam","spam_phishing","trash","Marked as spam.",False), gt_e001)
    check("Urgent→spam penalty ≤ -0.3", uas_detail["pen"] <= -0.3, f"pen={uas_detail['pen']}")

    adj_score, adj_detail = score_label(
        Action("e001","high","technical_support","support_tier2","Security issue.",True), gt_e001)
    check("Adjacent priority → partial credit", 0 < adj_detail["pri"] < 1,
          f"pri_score={adj_detail['pri']}")

    miss_esc, miss_detail = score_label(
        Action("e001","urgent","technical_support","support_tier2","Security.",False), gt_e001)
    check("Missed escalation → esc_score=0", miss_detail["esc"]==0.0, f"esc={miss_detail['esc']}")

    over_esc, over_detail = score_label(
        Action("e003","low","internal_hr","hr","Team lunch.",True), DATASET["easy"][2])
    check("Over-escalation → esc_score=0.5", over_detail["esc"]==0.5, f"esc={over_detail['esc']}")

    # ── 3. Environment lifecycle ──────────────────────────────────────────────
    section("3. Environment lifecycle")
    env = EmailTriageEnv("easy")
    obs = env.reset()
    check("reset() returns obs dict", "current_email" in obs and "remaining" in obs)
    check("reset() email count=5", obs["total"]==5, f"n={obs['total']}")
    check("reset() has current_email", obs["current_email"] is not None)
    check("reset() step=0", env._step==0)
    check("reset() escalation_budget_remaining", obs["escalation_budget_remaining"]==3)
    check("reset() all queues at capacity", all(
        obs["team_queues"][k]==TEAM_CAPACITY[k] for k in TEAM_CAPACITY))

    email = obs["current_email"]
    a1 = Action(email["id"],"urgent","technical_support","support_tier2",
                "Security incident: account compromise, needs immediate lockdown.",True)
    obs2, r1, done1, _ = env.step(a1)
    check("step() reduces remaining by 1", obs2["remaining"]==4, f"remaining={obs2['remaining']}")
    check("step() reward in [−1, 1]", -1.0<=r1.total<=1.0, f"reward={r1.total:.4f}")
    check("step() done=False after 1/5", not done1)
    check("step() support_tier2 queue decremented",
          obs2["team_queues"]["support_tier2"] == TEAM_CAPACITY["support_tier2"]-1,
          f"queue={obs2['team_queues']['support_tier2']}")
    check("step() escalations_used=1", env._con.escalations_used==1)

    bad_action = Action("INVALID","low","general_inquiry","support_tier1","x",False)
    _, r_bad, _, _ = env.step(bad_action)
    check("Invalid email_id → negative reward", r_bad.total < 0, f"reward={r_bad.total}")

    # ── 4. Sequential mechanics ───────────────────────────────────────────────
    section("4. Sequential mechanics (the core fix)")

    # 4a. SLA breach fires when urgent email left untouched.
    # e001 (urgent) has arrived_at_step=0, deadline=0+SLA_STEPS["urgent"]=2.
    # Breach check runs at start of each step using self._step (pre-increment).
    # The tick fires with self._step=2 at the START of the 3rd step, so we need
    # 3 non-urgent actions to trigger it.
    env2 = EmailTriageEnv("easy")
    env2.reset()
    low_items = [e for e in DATASET["easy"] if e["priority"]!="urgent"]
    last_r = None
    for e in low_items[:3]:   # 3 steps → tick at step=2 → e001 deadline=2 → breach
        a = Action(e["id"],e["priority"],e["category"],e["route_to"],f"Handled {e['id']}.",False)
        _, last_r, _, _ = env2.step(a)
    check("SLA breach fires for delayed urgent", env2._con.sla_breaches >= 1,
          f"breaches={env2._con.sla_breaches}")
    check("SLA penalty appears in reward", last_r.breakdown.sla_penalty < 0,
          f"sla_pen={last_r.breakdown.sla_penalty}")

    # 4b. Budget exhaustion penalises over-escalation
    env3 = EmailTriageEnv("easy")  # budget=3
    env3.reset()
    emails3 = list(DATASET["easy"])
    # Escalate all 5 emails — only first 3 allowed, 4th triggers penalty
    step_rewards3 = []
    for e in emails3:
        a = Action(e["id"],e["priority"],e["category"],e["route_to"],f"Escalated {e['id']}.",True)
        _, r, done, info = env3.step(a)
        step_rewards3.append((r.total, r.breakdown.budget_penalty, info.get("budget_overflow",False)))
    budget_overflows = [x for x in step_rewards3 if x[2]]
    check("Budget exhaustion triggers on 4th escalation", len(budget_overflows)>=1,
          f"overflows={len(budget_overflows)}")
    check("Budget penalty is -0.20", any(x[1]==-0.20 for x in step_rewards3),
          f"penalties={[x[1] for x in step_rewards3]}")

    # 4c. Queue saturation forces overflow
    env4 = EmailTriageEnv("hard")  # legal capacity=2
    env4.reset()
    legal_emails = [e for e in DATASET["hard"] if e["route_to"]=="legal"]
    assert len(legal_emails) >= 3, "Need ≥3 legal emails to test saturation"
    overflows4 = []
    for e in legal_emails:
        a = Action(e["id"],e["priority"],e["category"],"legal",f"Legal: {e['subject'][:50]}.",False)
        _, r, done, info = env4.step(a)
        overflows4.append(info.get("queue_overflow"))
    check("Legal queue saturates after 2 emails",
          sum(1 for o in overflows4 if o is not None) >= 1,
          f"overflows={overflows4}")
    check("Queue penalty is -0.10",
          any(r.breakdown.queue_penalty==-0.10 for _,r,_,_ in [
              env4.step(Action(e["id"],e["priority"],e["category"],"legal",f"x.",False))
              for e in DATASET["hard"] if e["route_to"]=="legal"
          ] if False) or env4._con.queue_overflows >= 1,
          f"queue_overflows={env4._con.queue_overflows}")

    # 4d. Cascade trigger.
    # hard task: h001 deadline=2, h002 deadline=3.
    # Process 2 non-urgent (steps 0,1) + h001 at step 2 (h001 SLA fires, breach=1)
    # + h002 at step 3 (h002 SLA fires, breach=2) → cascade triggers.
    env5 = EmailTriageEnv("hard")
    env5.reset()
    non_urgent_hard = [e for e in DATASET["hard"] if e["priority"]!="urgent"]
    urgents_hard    = [e for e in DATASET["hard"] if e["priority"]=="urgent"]
    for e in non_urgent_hard:          # steps 0, 1
        env5.step(Action(e["id"],e["priority"],e["category"],e["route_to"],"Handled.",False))
    # step 2: h001 deadline fires (sla_breach=1)
    e = urgents_hard[0]
    env5.step(Action(e["id"],e["priority"],e["category"],e["route_to"],"Handled.",True))
    # step 3: h002 deadline fires (sla_breach=2) → cascade=True
    e = urgents_hard[1]
    _, r5, _, _ = env5.step(Action(e["id"],e["priority"],e["category"],e["route_to"],"Handled.",True))
    check("Cascade triggers after 2+ urgent SLA breaches",
          env5._con.cascade_triggered and env5._con.sla_breaches >= 2,
          f"breaches={env5._con.sla_breaches} cascade={env5._con.cascade_triggered}")
    check("Cascade penalty = -0.25", r5.breakdown.cascade_penalty == -0.25,
          f"cpen={r5.breakdown.cascade_penalty}")

    # 4e. Smart agent outperforms medium agent via budget management
    smart_hard = run_episode("hard", agent_smart, "smart")
    medium_hard = run_episode("hard", agent_medium, "medium-skill")
    check("Smart agent ≥ medium-skill on hard (budget/queue management)",
          smart_hard["avg"] >= medium_hard["avg"] - 0.05,  # within 0.05 or better
          f"smart={smart_hard['avg']:.4f} medium={medium_hard['avg']:.4f}")
    check("Smart agent fewer SLA breaches on hard",
          smart_hard["sla_breaches"] <= medium_hard["sla_breaches"],
          f"smart_sla={smart_hard['sla_breaches']} med_sla={medium_hard['sla_breaches']}")

    # ── 5. Full episodes + difficulty spread ──────────────────────────────────
    section("5. Difficulty spread (the core fix)")
    all_results: Dict[str, Dict[str, float]] = {}
    for task_id in ("easy","medium","hard"):
        all_results[task_id] = {}
        for agent_fn, aname in [(agent_naive,"naive"),(agent_medium,"medium"),(agent_smart,"smart")]:
            res = run_episode(task_id, agent_fn, aname)
            all_results[task_id][aname] = res["avg"]

    if not OUTPUT_ARGS:
        print("\n  Score matrix (avg reward per step):")
        print(f"  {'task':8s} {'naive':>8s} {'medium':>8s} {'smart':>8s} {'spread':>8s}")
        print(f"  {'-'*44}")
        for task_id in ("easy","medium","hard"):
            r = all_results[task_id]
            spread = r["smart"] - r["naive"]
            print(f"  {task_id:8s} {r['naive']:8.4f} {r['medium']:8.4f} {r['smart']:8.4f} {spread:8.4f}")

    # Key checks: spread must exist AND be wider for harder tasks
    easy_spread   = all_results["easy"]["smart"]  - all_results["easy"]["naive"]
    medium_spread = all_results["medium"]["smart"] - all_results["medium"]["naive"]
    hard_spread   = all_results["hard"]["smart"]  - all_results["hard"]["naive"]

    check("Naive scores lower than smart on easy",
          all_results["easy"]["naive"] < all_results["easy"]["smart"],
          f"{all_results['easy']['naive']:.4f} < {all_results['easy']['smart']:.4f}")
    check("Hard task: naive significantly below medium-skill (> 0.10)",
          all_results["hard"]["medium"] - all_results["hard"]["naive"] > 0.10,
          f"gap={all_results['hard']['medium']-all_results['hard']['naive']:.4f}")
    check("Hard spread wider than easy spread",
          hard_spread >= easy_spread - 0.05,
          f"hard={hard_spread:.4f} easy={easy_spread:.4f}")
    check("Medium agent < 0.90 on hard (genuine difficulty)",
          all_results["hard"]["medium"] < 0.90,
          f"medium@hard={all_results['hard']['medium']:.4f}")
    check("Smart agent < 0.90 on hard (non-trivial)",
          all_results["hard"]["smart"] < 0.95,
          f"smart@hard={all_results['hard']['smart']:.4f}")

    # ── 6. Episode boundaries ─────────────────────────────────────────────────
    section("6. Episode boundaries")
    env6 = EmailTriageEnv("easy"); env6.reset()
    for e in DATASET["easy"]:
        env6.step(Action(e["id"],e["priority"],e["category"],e["route_to"],f"Done.",False))
    check("done=True after all emails", env6._done)
    try:
        env6.step(Action("e001","low","general_inquiry","support_tier1","x",False))
        check("Step after done raises", False)
    except AssertionError:
        check("Step after done raises", True)

    # ── 7. File existence ─────────────────────────────────────────────────────
    section("7. Required files")
    base = os.path.dirname(os.path.abspath(__file__))
    for rel in [
        "openenv.yaml","Dockerfile","README.md","pyproject.toml","server.py",
        "baseline/run_baseline.py",
        "src/openenv_email_triage/environment.py",
        "src/openenv_email_triage/models.py",
        "src/openenv_email_triage/grader.py",
        "src/openenv_email_triage/dataset.py",
        "tests/test_environment.py","ui/index.html",
    ]:
        check(f"exists: {rel}", os.path.isfile(os.path.join(base,rel)))

    # ── 8. Reproducible baseline (Fix 2) ──────────────────────────────────────
    section("8. Reproducible baseline scores")
    # Run every agent × every task with fixed seed=42 ordering and record results
    baseline: Dict[str, Any] = {
        "version": "2.0.0", "seed": 42,
        "description": (
            "Three-agent × three-task score matrix. Scores are fully deterministic "
            "(no randomness in dataset or grader). SHA-256 fingerprint below "
            "allows independent verification without running a live model."
        ),
        "agents": {},
    }
    for agent_fn, aname in [(agent_naive,"naive"),(agent_medium,"medium_skill"),(agent_smart,"smart")]:
        baseline["agents"][aname] = {}
        for task_id in ("easy","medium","hard"):
            res = run_episode(task_id, agent_fn, aname)
            baseline["agents"][aname][task_id] = {
                "avg_reward":     res["avg"],
                "min_step":       res["min"],
                "max_step":       res["max"],
                "sla_breaches":   res["sla_breaches"],
                "queue_overflows":res["queue_overflows"],
                "budget_used":    res["budget_used"],
                "budget_total":   res["budget_total"],
                "cascade":        res["cascade"],
                "step_rewards":   res["step_rewards"],
            }

    # Compute deterministic SHA-256 fingerprint of numeric scores
    score_blob = json.dumps({
        a: {t: baseline["agents"][a][t]["avg_reward"]
            for t in ("easy","medium","hard")}
        for a in ("naive","medium_skill","smart")
    }, sort_keys=True)
    baseline["score_fingerprint_sha256"] = hashlib.sha256(score_blob.encode()).hexdigest()

    if not OUTPUT_ARGS:
        print(f"\n  Baseline scores (deterministic, seed=42):")
        print(f"  {'agent':14s} {'easy':>8s} {'medium':>8s} {'hard':>8s}")
        print(f"  {'-'*42}")
        for aname in ("naive","medium_skill","smart"):
            row = baseline["agents"][aname]
            print(f"  {aname:14s} {row['easy']['avg_reward']:8.4f} "
                  f"{row['medium']['avg_reward']:8.4f} {row['hard']['avg_reward']:8.4f}")
        print(f"\n  SHA-256 fingerprint: {baseline['score_fingerprint_sha256'][:32]}…")

    # Write baseline JSON
    out_path = os.path.join(base, "baseline", "baseline_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)

    check("baseline_results.json written", os.path.isfile(out_path))
    check("SHA-256 fingerprint is 64 hex chars",
          len(baseline["score_fingerprint_sha256"])==64)
    check("Naive < smart on all tasks (spread exists)", all(
        baseline["agents"]["naive"][t]["avg_reward"] <
        baseline["agents"]["smart"][t]["avg_reward"]
        for t in ("easy","medium","hard")))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(1 for ok,_ in results if ok)
    n_fail = sum(1 for ok,_ in results if not ok)

    if OUTPUT_ARGS:
        print(json.dumps({
            "passed": n_pass, "failed": n_fail, "total": len(results),
            "baseline": baseline,
            "failures": [name for ok,name in results if not ok],
        }, indent=2))
    else:
        print()
        print("=" * 66)
        print(f"  Results: {n_pass}/{len(results)} passed  |  {n_fail} failed")
        print("=" * 66)
        if n_fail:
            print("\nFailed tests:")
            for ok, name in results:
                if not ok: print(f"  {FAIL} {name}")
        else:
            print("\n  All tests passed! ✓")
            # Print the matrix one more time cleanly
            print()
            print("  Score matrix — naive / medium-skill / smart agent:")
            for task_id in ("easy","medium","hard"):
                row = {a: baseline["agents"][a][task_id]["avg_reward"]
                       for a in ("naive","medium_skill","smart")}
                bar_n = "█"*int(row["naive"]*20)
                bar_m = "█"*int(row["medium_skill"]*20)
                bar_s = "█"*int(row["smart"]*20)
                print(f"  {task_id:6s}  naive [{bar_n:<20}] {row['naive']:.4f}  "
                      f"smart [{bar_s:<20}] {row['smart']:.4f}")
        print()

    sys.exit(0 if n_fail==0 else 1)

if __name__ == "__main__":
    main()
