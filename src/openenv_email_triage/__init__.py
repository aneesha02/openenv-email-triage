"""
openenv-email-triage
====================
An OpenEnv-compliant environment that simulates real-world email triage.
Agents must classify, prioritize, and route incoming emails across three
difficulty levels: easy, medium, and hard.
"""

from .environment import EmailTriageEnv
from .models import Action, Observation, Reward, EnvironmentState, Priority, Category, RouteTo
from .grader import score_action, grade_episode

__all__ = [
    "EmailTriageEnv",
    "Action",
    "Observation",
    "Reward",
    "EnvironmentState",
    "Priority",
    "Category",
    "RouteTo",
    "score_action",
    "grade_episode",
]

__version__ = "1.0.0"
