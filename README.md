---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: OpenEnv email triage benchmark
tags:
  - openenv
  - agents
  - evaluation
  - email-triage
---

# Email Triage OpenEnv

A real-world agent benchmark for corporate email triage.

The environment simulates the kind of inbox-workflow a human operator, support lead, or operations agent actually performs: reading incoming emails, classifying intent, choosing a routing destination, deciding whether to escalate, and adapting to limited team capacity and SLA pressure over a trajectory of decisions.

## Why this environment exists

Most email triage systems are evaluated as isolated classification problems. This environment goes further by making triage sequential:

- escalation is budgeted,
- routing destinations have finite capacity,
- urgent messages can breach SLA deadlines if ignored too long,
- and early actions affect later options.

That makes the task useful for evaluating agents that must plan, not just classify.

## Environment interface

The environment follows the OpenEnv-style interface:

- `reset()` returns the initial observation for the selected task.
- `step(action)` returns `(observation, reward, done, info)`.
- `state()` returns the current internal state.
- `Observation`, `Action`, and `Reward` are typed Pydantic models.

## Observation space

Each observation exposes the current inbox state and the sequential constraints needed to plan correctly.

Important fields include:

- `inbox`: full list of emails still relevant to the episode
- `current_email`: the email expected for the current step
- `processed`: IDs of emails already handled
- `step_number`: current episode step
- `total_emails`: total emails in the episode
- `remaining`: emails left to process
- `escalation_budget_remaining`: remaining `flag_review` capacity
- `team_queue_remaining`: remaining capacity per routing destination
- `active_sla_warnings`: emails close to or at deadline
- `sla_breaches_so_far`: number of SLA breaches already triggered
- `cascade_active`: whether the one-time cascade penalty has already been triggered

## Action space

Each action contains:

- `email_id`: the email being handled
- `priority`: one of `urgent`, `high`, `medium`, `low`, `spam`
- `category`: one of:
  - `customer_complaint`
  - `billing_inquiry`
  - `technical_support`
  - `sales_lead`
  - `internal_hr`
  - `legal_compliance`
  - `spam_phishing`
  - `general_inquiry`
- `route_to`: one of:
  - `support_tier1`
  - `support_tier2`
  - `billing`
  - `sales`
  - `legal`
  - `hr`
  - `management`
  - `trash`
  - `archive`
- `summary`: a short natural-language summary
- `flag_review`: whether the message should consume escalation budget
- `reasoning`: free-form reasoning text, not scored directly

## Tasks

### Easy
A small inbox with forgiving SLAs and extra escalation slack. A competent agent should score well with straightforward triage.

### Medium
A larger inbox with tighter SLA pressure and exactly enough escalation budget to matter. The agent must balance correct classification with order of processing.

### Hard
A high-stakes inbox where escalation budget is insufficient for all truly urgent items, and team capacity can saturate mid-episode. The agent must choose carefully which emails to escalate and which messages to process first.

## Reward design

The reward is shaped across the full episode rather than only at the end.

It combines:

- label correctness for priority, category, routing, summary quality, and escalation choice
- SLA penalties for leaving urgent emails too long
- queue overflow penalties when routing saturates a destination
- budget overflow penalties when escalation budget is exhausted
- special penalties for spam-related mistakes and urgent spam misclassification

Rewards are designed to produce a meaningful trajectory signal in the range `[-1.0, 1.0]`.


## Output Format

```text id="fmt1"
[START] task=<task_name> env=email-triage-v1 model=<model_name>
[STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
```
## Environment Variables

You must set the following:

- HF_TOKEN: Hugging Face API token

### Local run
```bash
export HF_TOKEN=your_token_here
uv run server

## Set environment variables
Windows (PowerShell)
$env:HF_TOKEN="your_token_here"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:API_BASE_URL="https://router.huggingface.co/v1"

## Generate lock file (required)
uv lock

## Run the server
uv run server


## Test the API


curl -X POST http://localhost:7860/run \
  -H "Content-Type: application/json" \
  -d '{"task":"easy"}'

## Validate the environment
openenv validate

## Docker

```bash id="fmt2"
docker build -t email-triage-env .
docker run -p 7860:7860 -e HF_TOKEN=$env:HF_TOKEN email-triage-env
```

## Accessing the app

- Local: http://127.0.0.1:7860