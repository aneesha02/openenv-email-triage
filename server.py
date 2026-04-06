"""
FastAPI server exposing the EmailTriageEnv via HTTP.
Also serves the interactive web UI for manual exploration.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenv_email_triage import EmailTriageEnv, Action
from openenv_email_triage.grader import grade_episode

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An OpenEnv-compliant email triage environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session store (single-user; extend to Redis for multi-user) ──
_sessions: Dict[str, EmailTriageEnv] = {}


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed:    Optional[int] = None


class StepRequest(BaseModel):
    session_id: str
    action:     Dict[str, Any]


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "email-triage-v1"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Basic Email Triage",
                "difficulty": "easy",
                "n_emails": 5,
                "expected_score_range": [0.75, 1.0],
            },
            {
                "id": "medium",
                "name": "Mixed Inbox Triage",
                "difficulty": "medium",
                "n_emails": 8,
                "expected_score_range": [0.55, 0.80],
            },
            {
                "id": "hard",
                "name": "High-Stakes Inbox Triage",
                "difficulty": "hard",
                "n_emails": 10,
                "expected_score_range": [0.40, 0.70],
            },
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest):
    session_id = f"{req.task_id}-{os.urandom(4).hex()}"
    env = EmailTriageEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _sessions[session_id] = env
    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
    }


@app.post("/step")
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    try:
        action = Action(**req.action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")

    obs, reward, done, info = env.step(action)

    if done:
        # Clean up session
        del _sessions[req.session_id]

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state/{session_id}")
def state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state().model_dump()


@app.get("/action-space")
def action_space():
    from openenv_email_triage.models import Priority, Category, RouteTo
    return {
        "email_id":    "string — ID from current_email.header.email_id",
        "priority":    [p.value for p in Priority],
        "category":    [c.value for c in Category],
        "route_to":    [r.value for r in RouteTo],
        "summary":     "string, max 280 chars",
        "flag_review": "boolean",
        "reasoning":   "string (not scored)",
    }


# ─── Interactive UI ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    with open("/app/ui/index.html") as f:
        return f.read()

import subprocess, sys

@app.post("/grader")
def grader(req: StepRequest):
    """Returns grader score after an episode — required by OpenEnv spec."""
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    s = env.state()
    if not s.actions_taken:
        return {"overall_score": 0.0, "num_emails": 0, "message": "No actions taken yet"}
    scores = grade_episode(s.actions_taken)
    return scores

@app.get("/baseline")
def baseline():
    """Trigger rule-based baseline and return scores for all 3 tasks."""
    from openenv_email_triage.grader import grade_episode
    from openenv_email_triage.dataset import EASY_EMAILS, MEDIUM_EMAILS, HARD_EMAILS
    results = {}
    for task_id, dataset in [("easy", EASY_EMAILS), ("medium", MEDIUM_EMAILS), ("hard", HARD_EMAILS)]:
        env = EmailTriageEnv(task_id=task_id)
        env.reset()
        # Rule-based naive baseline
        actions_log = []
        while not env.is_done:
            obs = env._make_observation()
            if not obs.current_email:
                break
            email = obs.current_email
            # Simple keyword heuristic
            subject = email.header.subject.lower()
            body = email.body.lower()
            priority = "medium"
            category = "general_inquiry"
            route_to = "support_tier1"
            flag_review = False
            if any(w in subject+body for w in ["urgent", "legal", "lawsuit", "breach", "outage"]):
                priority = "urgent"; route_to = "management"; flag_review = True
            elif any(w in subject+body for w in ["billing", "invoice", "payment"]):
                category = "billing_inquiry"; route_to = "billing"
            elif any(w in subject+body for w in ["spam", "phishing", "lottery"]):
                priority = "spam"; route_to = "trash"
            from openenv_email_triage.models import Action, Priority, Category, RouteTo
            action = Action(
                email_id=email.header.email_id,
                priority=Priority(priority),
                category=Category(category),
                route_to=RouteTo(route_to),
                summary=email.header.subject[:280],
                flag_review=flag_review,
            )
            _, reward, done, _ = env.step(action)
            actions_log.append(action.model_dump())
        results[task_id] = grade_episode(actions_log) if actions_log else {"overall_score": 0.0}
    return {"baseline_scores": results, "model": "rule_based_heuristic"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
