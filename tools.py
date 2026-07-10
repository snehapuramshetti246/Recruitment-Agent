"""
tools.py — The four tools for the TechVest Recruitment Agent.

Tool 1: parse_resume    — READ   — LLM structured extraction → CandidateProfile
Tool 2: score_candidate — READ   — LLM rubric scoring → ScoreCard
Tool 3: check_availability — READ — Mock availability → list[Slot]
Tool 4: propose_interview  — WRITE — Action tool (NEVER fires without human approval)

Each tool is a plain Python function with Pydantic-typed I/O, decorated with
@tool for LangGraph/LangChain. All LLM calls use OpenRouter.
"""

from __future__ import annotations

import json
import random
from typing import Annotated

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from config import (
    GITHUB_TOKEN,
    GITHUB_MODELS_BASE_URL,
    LLM_MODEL,
    MOCK_SLOTS,
    RUBRIC,
)
from guardrails import sanitise_resume, assert_human_approved
from models import (
    CandidateProfile,
    CriterionScore,
    ScoreCard,
    Slot,
    Confirmation,
)


# ---------------------------------------------------------------------------
# LLM client (shared across tools)
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance pointing at GitHub Models."""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=GITHUB_TOKEN,
        base_url=GITHUB_MODELS_BASE_URL,
        temperature=0.0,  # deterministic for scoring
        max_retries=2,
        max_tokens=2000,
    )


# ---------------------------------------------------------------------------
# Tool 1 — parse_resume (READ)
# ---------------------------------------------------------------------------

@tool
def parse_resume(resume_text: str) -> dict:
    """
    Parse a candidate résumé into a structured CandidateProfile.

    Uses LLM structured extraction. The résumé text is first scanned and
    sanitised by the injection-defence guardrail before being sent to the LLM.

    Args:
        resume_text: Raw text of the candidate's résumé.

    Returns:
        dict representation of CandidateProfile.
    """
    # GUARDRAIL 3 — sanitise before sending to LLM
    sanitised_text, injection_detected, injection_details = sanitise_resume(resume_text)

    llm = _get_llm()

    system_prompt = """You are a résumé parser for a technical hiring process.
Extract information ONLY from the résumé text provided. Do NOT infer,
assume, or fill in information that is not explicitly stated.

If a field is not mentioned, use an empty string or 0.

Return a JSON object with EXACTLY these fields:
{
  "name": "Full name of the candidate",
  "skills": ["list", "of", "all", "technical", "skills", "mentioned"],
  "years_experience": 0.0,
  "education": "Degree, major, institution",
  "projects": ["Project 1 one-line summary", "Project 2 one-line summary"],
  "llm_api_experience": false,
  "has_production_code": false,
  "raw_text_excerpt": "Most relevant 200-char excerpt from the résumé"
}

Rules:
- years_experience: sum internship months / 12, round to 1 decimal. 0 if none.
- llm_api_experience: true only if candidate made actual API calls (not just chatbot use).
- has_production_code: true only if candidate deployed or shipped code to production.
- Return ONLY valid JSON, no extra text."""

    user_prompt = f"Parse this résumé:\n\n{sanitised_text}"

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    raw_content = response.content.strip()

    # Strip markdown code fences if present
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()

    profile_data = json.loads(raw_content)

    # Validate via Pydantic
    profile = CandidateProfile(**profile_data)
    result = profile.model_dump()

    # Attach injection metadata so the agent can log it
    result["_injection_detected"] = injection_detected
    result["_injection_details"] = injection_details

    return result


# ---------------------------------------------------------------------------
# Tool 2 — score_candidate (READ)
# ---------------------------------------------------------------------------

@tool
def score_candidate(profile_json: str, rubric_json: str) -> dict:
    """
    Score a candidate against the hiring rubric using LLM reasoning.

    Every criterion score must be backed by a specific evidence line from the
    résumé. The LLM is instructed that ungrounded scores are invalid.

    Args:
        profile_json: JSON string of a CandidateProfile dict.
        rubric_json:  JSON string of the rubric (from rubric.json).

    Returns:
        dict representation of ScoreCard.
    """
    llm = _get_llm()

    system_prompt = """You are an objective hiring evaluator for a Junior AI Engineer role.
You will score a candidate against a rubric. Your scores MUST be grounded in evidence.

RULES:
1. Score each criterion on the 0-5 scale defined in the rubric.
2. For each score > 0, you MUST quote a specific line, section, or project from the
   candidate profile as evidence. No evidence = score of 0.
3. Do NOT consider the candidate's name, gender, age, college prestige, GPA, or any
   attribute not in the rubric. Only the five criteria matter.
4. Be honest. A strong candidate should score high; a weak one should score low.
5. Scores should reflect actual demonstrated capability, not potential.

Return a JSON object with EXACTLY this structure:
{
  "candidate": "candidate name",
  "criteria": [
    {
      "criterion": "criterion_id",
      "criterion_name": "Human-readable name",
      "weight": 0.25,
      "score": 4,
      "evidence": "Specific line or section from profile that justifies this score"
    }
  ],
  "weighted_total": 3.8
}

weighted_total = sum(weight * score) for all criteria. Compute it yourself.
Return ONLY valid JSON."""

    user_prompt = (
        f"Candidate Profile:\n{profile_json}\n\n"
        f"Scoring Rubric:\n{rubric_json}"
    )

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    raw_content = response.content.strip()
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()

    scorecard_data = json.loads(raw_content)

    # Recompute weighted_total ourselves to ensure accuracy
    total = 0.0
    for c in scorecard_data.get("criteria", []):
        total += c.get("weight", 0.0) * c.get("score", 0)
    scorecard_data["weighted_total"] = round(total, 3)

    # Validate
    scorecard = ScoreCard(**scorecard_data)
    return scorecard.model_dump()


# ---------------------------------------------------------------------------
# Tool 3 — check_availability (READ / MOCK)
# ---------------------------------------------------------------------------

@tool
def check_availability(candidate_name: str, week: str) -> list[dict]:
    """
    Return available interview slots for a candidate in a given week.

    This is a mock tool — in production it would call a calendar API.
    Returns 3 randomly chosen slots from the mock slot pool.

    Args:
        candidate_name: Candidate's name (used for deterministic seeding).
        week:           ISO week string, e.g. "2025-W29".

    Returns:
        List of Slot dicts.
    """
    # Deterministic shuffle per candidate so results are reproducible
    rng = random.Random(hash(candidate_name + week) % (2**32))
    all_slots = MOCK_SLOTS.get("default", [])
    chosen = rng.sample(all_slots, min(3, len(all_slots)))

    slots = [Slot(**s) for s in chosen]
    return [s.model_dump() for s in slots]


# ---------------------------------------------------------------------------
# Tool 4 — propose_interview (WRITE / ACTION — requires human approval)
# ---------------------------------------------------------------------------

@tool
def propose_interview(candidate_name: str, slot_json: str, approved: bool = False) -> dict:
    """
    Propose (and, after human approval, confirm) an interview slot.

    THIS TOOL IS AN ACTION TOOL. It must NEVER be called with approved=True
    unless a human has explicitly clicked "Approve" in the UI.
    The LangGraph schedule_node enforces this via interrupt().

    Args:
        candidate_name: Candidate's full name.
        slot_json:      JSON string of a Slot dict.
        approved:       Must be True only after explicit human approval.

    Returns:
        dict representation of Confirmation.
    """
    slot_data = json.loads(slot_json) if isinstance(slot_json, str) else slot_json
    slot = Slot(**slot_data)

    # Default to pending — human gate is enforced by the graph, but we
    # add a programmatic check here as a belt-and-suspenders safeguard.
    status = "confirmed" if approved else "pending_approval"

    if approved:
        # Belt-and-suspenders: the assert will raise if somehow called wrong
        assert_human_approved(status, candidate_name)

    import uuid
    conf = Confirmation(
        candidate=candidate_name,
        slot=slot,
        status=status,
        confirmation_id=str(uuid.uuid4())[:8] if approved else "",
    )
    return conf.model_dump()


# ---------------------------------------------------------------------------
# Tool registry (for agent binding)
# ---------------------------------------------------------------------------

READ_TOOLS = [parse_resume, score_candidate, check_availability]
ACTION_TOOLS = [propose_interview]
ALL_TOOLS = READ_TOOLS + ACTION_TOOLS
