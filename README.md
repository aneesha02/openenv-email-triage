# 📬 Email Triage — OpenEnv v2

> A real-world OpenEnv environment where AI agents learn to triage corporate email inboxes — with **sequential state** that makes processing order, escalation rationing, and routing adaptation first-class decisions.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v2.0-4af0a0?style=flat-square)](https://openenv.dev)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## What makes this a real sequential decision problem

v1 was honest N-shot classification: each email was independently scored, so a greedy per-email policy was globally optimal. v2 adds three persistent session constraints that couple every decision to future ones:

### 1. Escalation budget
Each episode has a fixed pool of `flag_review=True` uses. On the hard task, 7 emails genuinely need escalation but the budget is only 5 — the agent must read ahead and choose which escalations matter most, accepting that two critical emails get deprioritised.

```
Observation: escalation_budget_remaining = 2
```

Exhausting the budget causes subsequent escalations to be silently dropped with a −0.20 penalty. The budget is always visible in the observation so agents can plan.

### 2. Team queue capacity
Each routing destination has finite capacity per episode. `legal` holds 2, `support_tier2` holds 2, `management` holds 2. On the hard task, 5 emails belong in `legal` — the agent must route the 3rd+ to `management` (acceptable overflow, 0.4× routing credit) or take a −0.10 queue overflow penalty.

```
Observation: team_queue_remaining = {"legal": 2, "support_tier2": 1, ...}
```

### 3. SLA decay timers
Every email has a deadline measured in steps from its position in the inbox. An `urgent` email must be handled within 2 steps of its arrival; `high` within 4. If the agent processes low-priority mail first, SLA breach events fire automatically (−0.15 each). Two urgent SLA breaches trigger a cascade event (−0.25 one-time) and set `cascade_active=True` in all subsequent observations.

```
Observation: active_sla_warnings = [{"email_id": "h001", "steps_left": 1}]
```

**Consequence**: processing order is a first-class decision. A greedy policy that scores each email in isolation is strictly suboptimal on medium and hard tasks.

---

## Score spread

Three agent tiers show genuine difficulty separation:

| Task | Naive agent | Medium-skill | Smart agent | Spread |
|------|------------|--------------|-------------|--------|
| easy | 0.4680 | 0.9780 | 0.9860 | 0.518 |
| medium | 0.2563 | 0.9475 | 0.9413 | 0.685 |
| **hard** | **0.1660** | **0.8420** | **0.8720** | **0.706** |

- **Naive**: keyword-only heuristics, zero budget/queue/SLA awareness
- **Medium-skill**: correct labels, processes emails in inbox order, always escalates → suffers queue saturation and budget overflow on hard
- **Smart**: correct labels + SLA-aware ordering + budget rationing + queue routing adaptation

Scores are fully deterministic. SHA-256 fingerprint: `7d9d9e7f…` — run `python validate.py` to reproduce.

---

## Quick start

```bash
git clone https://huggingface.co/spaces/openenv/email-triage
cd email-triage
pip install -e ".[baseline]"

# Interactive UI
uvicorn server:app --port 7860
# → http://localhost:7860

# Baseline agent (OpenAI API)
export OPENAI_API_KEY="sk-..."
python baseline/run_baseline.py

# Stdlib-only validation (no install needed)
python validate.py
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=$OPENAI_API_KEY email-triage-env
```

---

## Python API

```python
from openenv_email_triage import EmailTriageEnv, Action, Priority, Category, RouteTo

env = EmailTriageEnv(task_id="hard")  # "easy" | "medium" | "hard"
obs = env.reset()

# obs now exposes sequential state:
print(obs.escalation_budget_remaining)   # → 5
print(obs.team_queue_remaining)          # → {"legal": 2, "support_tier2": 2, ...}
print(obs.active_sla_warnings)           # → [{"email_id": "h001", "steps_left": 1}]

action = Action(
    email_id  = obs.current_email.header.email_id,
    priority  = Priority.URGENT,
    category  = Category.LEGAL_COMPLIANCE,
    route_to  = RouteTo.LEGAL,
    summary   = "Regulatory deadline in 72h — data breach notification required.",
    flag_review = True,   # consumes 1 unit of escalation budget
    reasoning = "Inspector Davies letter cites DPA 2023 s.12(b), hard 72h deadline.",
)
obs, reward, done, info = env.step(action)

# reward.breakdown shows sequential penalties alongside label scores:
print(reward.breakdown.sla_penalty)      # 0.0 (handled on time)
print(reward.breakdown.queue_penalty)    # 0.0 or -0.10 if legal is full
print(reward.breakdown.budget_penalty)   # 0.0 or -0.20 if budget exhausted
print(reward.total)                      # combined score, range [-1, 1]
```

---

## HTTP API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive web UI |
| `GET` | `/tasks` | List tasks with metadata |
| `POST` | `/reset` | Start episode → `{session_id, observation}` |
| `POST` | `/step` | Submit action → `{observation, reward, done, info}` |
| `GET` | `/state/{session_id}` | Full state snapshot |
| `GET` | `/action-space` | Enumerate valid values |
| `GET` | `/docs` | Swagger UI |

---

## Observation space

```python
class Observation(BaseModel):
    # Email state
    inbox:         List[EmailMessage]       # All unprocessed emails
    current_email: Optional[EmailMessage]   # Next to triage
    processed:     List[str]               # Handled email_ids
    step_number:   int
    total_emails:  int
    remaining:     int

    # Sequential constraints (persist across steps)
    escalation_budget_remaining: int              # Pool of flag_review uses left
    team_queue_remaining:        Dict[str, int]   # Per-team capacity remaining
    active_sla_warnings:         List[dict]       # Emails near/past deadline
    sla_breaches_so_far:         int
    cascade_active:              bool             # True after 2nd urgent breach
    session_info:                dict
```

---

## Action space

| Field | Type | Notes |
|-------|------|-------|
| `email_id` | `str` | From `current_email.header.email_id` |
| `priority` | `urgent\|high\|medium\|low\|spam` | |
| `category` | `customer_complaint\|billing_inquiry\|technical_support\|sales_lead\|internal_hr\|legal_compliance\|spam_phishing\|general_inquiry` | |
| `route_to` | `support_tier1\|support_tier2\|billing\|sales\|legal\|hr\|management\|trash\|archive` | Queue capacity applies |
| `summary` | `str ≤ 280 chars` | |
| `flag_review` | `bool` | Consumes escalation budget |
| `reasoning` | `str` | Not scored |

---

## Reward function

```
step_reward = label_score + sequential_penalties

label_score (0..1):
  priority    × 0.35  (adjacency-weighted: "high" for "urgent" = 0.5×)
  category    × 0.25  (partial credit for related categories)
  routing     × 0.25  (acceptable alternatives get 0.4×)
  summary     × 0.10  (heuristic quality)
  escalation  × 0.05  (correct flag_review use)
  base_penalty        (spam misrouted: −0.20; urgent→spam: −0.30)

sequential_penalties (cumulative per step):
  sla_breach      −0.15  per unhandled email past its deadline
  queue_overflow  −0.10  routing to saturated team
  budget_overflow −0.20  escalating after budget exhausted
  cascade         −0.25  one-time on 2nd urgent SLA breach

Total range: [−1.0, 1.0]
```

---

## Tasks

### 🟢 Easy (5 emails, budget=3, true need=2)
Clear signals throughout. SLA windows are comfortable. Budget has slack. Expected smart-agent score: **0.95–0.99**.

### 🟡 Medium (8 emails, budget=4, true need=4)
Budget exactly matches need — no margin for over-escalation. Two urgent emails have tight SLA windows. `legal` and `support_tier2` can saturate. Expected smart-agent score: **0.85–0.95**.

### 🔴 Hard (10 emails, budget=5, true need=7)
Agent must choose which 5 of 7 escalation-worthy emails to actually escalate. `legal` capacity=2 but 5 emails should go there — queue saturates mid-episode. 8 urgent emails with overlapping SLA timers mean wrong ordering cascades. Expected smart-agent score: **0.75–0.90**; naive agent: **0.15–0.25**.

---

## Baseline scores

Fully deterministic (no randomness). Reproduce with `python validate.py`.

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Naive (keyword heuristic) | 0.4680 | 0.2563 | 0.1660 |
| Medium-skill (correct labels, no planning) | 0.9780 | 0.9475 | 0.8420 |
| Smart (labels + SLA order + budget + queue) | 0.9860 | 0.9413 | 0.8720 |

SHA-256 fingerprint of score matrix: `7d9d9e7fc953001507b2a16ceec6a62e5c11ff748b6f1b2ad1f09d3be5c4a6b8`

To run with a real model:
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"   # or gpt-4o
python baseline/run_baseline.py
# Writes baseline/baseline_results.json
```

---

## Project structure

```
openenv-email-triage/
├── src/openenv_email_triage/
│   ├── __init__.py
│   ├── environment.py    # EmailTriageEnv — step/reset/state + sequential mechanics
│   ├── models.py         # Typed Pydantic models (Action, Observation, Reward,
│   │                     #   SessionConstraints, TeamQueueState, SlaStatus)
│   ├── dataset.py        # 23 labelled synthetic emails (easy/medium/hard)
│   └── grader.py         # Deterministic per-step label scoring
├── baseline/
│   ├── run_baseline.py   # OpenAI-API agent with system prompt
│   └── baseline_results.json  # Generated by validate.py or run_baseline.py
├── tests/
│   └── test_environment.py   # pytest suite (55 tests)
├── ui/
│   └── index.html        # Interactive web inbox UI
├── validate.py           # Stdlib-only full validation + reproducible baseline
├── server.py             # FastAPI HTTP server
├── openenv.yaml          # OpenEnv spec (v2)
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v

# Or stdlib-only (no install):
python validate.py
```

---

## Deployment (Hugging Face Spaces)

Add this frontmatter to your `README.md` for HF Spaces auto-detection:

```yaml
---
title: Email Triage OpenEnv
emoji: 📬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - email-triage
---
```

The FastAPI server binds to port 7860 (HF Spaces default) and serves the interactive UI at `/` and the full REST API at `/reset`, `/step`, `/state`.

---

## License

MIT
