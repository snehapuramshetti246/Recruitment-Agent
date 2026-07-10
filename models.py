"""
models.py — Pydantic v2 schemas for the TechVest Recruitment Agent.

All tool inputs/outputs and LangGraph state sub-objects are typed here.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Candidate profile — output of parse_resume tool
# ---------------------------------------------------------------------------

class CandidateProfile(BaseModel):
    """Structured extraction of a candidate résumé."""
    name: str = Field(description="Candidate's full name")
    skills: list[str] = Field(description="All technical skills mentioned in the résumé")
    years_experience: float = Field(
        description="Total years of relevant professional/internship experience (0 if none)",
        ge=0.0,
    )
    education: str = Field(description="Highest education qualification and institution")
    projects: list[str] = Field(description="List of project names / one-line summaries")
    llm_api_experience: bool = Field(
        default=False,
        description="True if the candidate has used any LLM API in code",
    )
    has_production_code: bool = Field(
        default=False,
        description="True if the candidate has shipped or deployed production code",
    )
    raw_text_excerpt: str = Field(
        default="",
        description="Brief excerpt (≤200 chars) of the most relevant résumé section",
    )


# ---------------------------------------------------------------------------
# Scoring — output of score_candidate tool
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    """Score for a single rubric criterion."""
    criterion: str = Field(description="Criterion id from the rubric")
    criterion_name: str = Field(description="Human-readable criterion name")
    weight: float = Field(description="Weight from rubric (0–1)", ge=0.0, le=1.0)
    score: int = Field(description="Score 0–5 per rubric scale", ge=0, le=5)
    evidence: str = Field(
        description="Verbatim or paraphrased résumé line that justifies the score. "
                    "'No evidence found' if score is 0."
    )


class ScoreCard(BaseModel):
    """Full scored rubric for one candidate."""
    candidate: str = Field(description="Candidate's name")
    criteria: list[CriterionScore] = Field(description="One entry per rubric criterion")
    weighted_total: float = Field(
        description="Sum of (weight × score) for all criteria. Max = 5.0", ge=0.0, le=5.0
    )


# ---------------------------------------------------------------------------
# Availability — output of check_availability tool
# ---------------------------------------------------------------------------

class Slot(BaseModel):
    """A single available interview slot."""
    day: str = Field(description="Day of the week, e.g. 'Monday'")
    date: str = Field(description="Date string, e.g. '2025-07-14'")
    time: str = Field(description="Time string in 24h, e.g. '10:00'")
    duration_minutes: int = Field(default=60, description="Interview duration in minutes")


# ---------------------------------------------------------------------------
# Confirmation — output of propose_interview tool
# ---------------------------------------------------------------------------

class Confirmation(BaseModel):
    """Outcome of proposing an interview slot."""
    candidate: str = Field(description="Candidate's name")
    slot: Slot = Field(description="The proposed slot")
    status: Literal["pending_approval", "confirmed", "rejected"] = Field(
        default="pending_approval",
        description="'confirmed' only after explicit human approval",
    )
    confirmation_id: str = Field(
        default="",
        description="Unique ID for this confirmation (assigned on confirm)",
    )


# ---------------------------------------------------------------------------
# Shortlist entry — final per-candidate decision
# ---------------------------------------------------------------------------

class ShortlistEntry(BaseModel):
    """The agent's final recommendation for one candidate."""
    candidate: str
    verdict: Literal["INTERVIEW", "HOLD", "NOT_A_FIT"]
    weighted_score: float = Field(ge=0.0, le=5.0)
    justification: str = Field(
        description="Paragraph citing specific résumé evidence for the verdict"
    )
    scorecard: ScoreCard
    proposed_slot: Optional[Slot] = None
    confirmation: Optional[Confirmation] = None


# ---------------------------------------------------------------------------
# Trajectory — the thought/action/observation log
# ---------------------------------------------------------------------------

class TrajectoryStep(BaseModel):
    """One logged step in the agent's reasoning trace."""
    step_number: int
    thought: str = Field(description="Agent's reasoning about what to do next")
    action: str = Field(description="Tool name called, or 'decide' / 'complete'")
    action_args: dict = Field(default_factory=dict, description="Arguments passed to the tool")
    observation: str = Field(description="What the tool returned (stringified)")
    guardrail_triggered: Optional[str] = Field(
        default=None,
        description="Name of the guardrail that fired on this step, if any",
    )


# ---------------------------------------------------------------------------
# Guardrail flags — live status tracked in state
# ---------------------------------------------------------------------------

class GuardrailFlags(BaseModel):
    """Live guardrail status for the current run."""
    injection_detected: bool = False
    injection_details: str = ""
    fairness_pass: Optional[bool] = None   # None = not yet tested
    step_cap: int = 25
    steps_used: int = 0
    human_gate_status: Literal["armed", "waiting_approval", "approved", "rejected"] = "armed"
    audit_log_path: str = "decisions.json"


# ---------------------------------------------------------------------------
# Run statistics — collected at end of run
# ---------------------------------------------------------------------------

class RunStats(BaseModel):
    """Summary stats for one full agent run."""
    total_steps: int = 0
    tool_calls: dict[str, int] = Field(default_factory=dict)
    run_duration_seconds: float = 0.0
    timestamp: str = ""
    candidates_processed: int = 0
