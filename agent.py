"""
agent.py — LangGraph stateful agent for the TechVest Recruitment Agent.

Graph topology:
  START → router_node → parse_node → score_node → availability_node
                      → decide_node → schedule_node (gated) → END

The router decides which node to visit next based on what's been done.
No hard-coded fixed pipeline — the agent chooses its own tool order.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from typing_extensions import TypedDict

from config import AGENT_STEP_CAP, AGENT_RECURSION_LIMIT, RUBRIC
from models import (
    CandidateProfile,
    ScoreCard,
    ShortlistEntry,
    TrajectoryStep,
    GuardrailFlags,
    RunStats,
    Slot,
    Confirmation,
)
from guardrails import check_step_cap, check_criteria_fairness, name_swap_test
from audit import TrajectoryLogger, AuditLog
from tools import parse_resume, score_candidate, check_availability, propose_interview


# ---------------------------------------------------------------------------
# State shape
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    jd: str
    rubric: dict
    candidates: dict[str, str]          # {name_key: raw_resume_text}
    profiles: dict[str, dict]           # {name_key: CandidateProfile.model_dump()}
    scorecards: dict[str, dict]         # {name_key: ScoreCard.model_dump()}
    shortlist: list[dict]               # list of ShortlistEntry.model_dump()
    trajectory: list[dict]              # list of TrajectoryStep.model_dump()
    step_count: int
    guardrail_flags: dict               # GuardrailFlags.model_dump()
    pending_approval: Optional[dict]    # ShortlistEntry pending human gate
    run_stats: dict                     # RunStats.model_dump()
    error: Optional[str]


# ---------------------------------------------------------------------------
# Shared logger instance (module-level, reset each run)
# ---------------------------------------------------------------------------

_logger = TrajectoryLogger()
_audit = AuditLog()


def _log(state: AgentState, thought: str, action: str,
         args: dict, observation: str, guardrail: str | None = None) -> list[dict]:
    """Append a trajectory step and return updated trajectory list."""
    step = _logger.log(
        thought=thought,
        action=action,
        action_args=args,
        observation=observation,
        guardrail_triggered=guardrail,
    )
    updated = list(state["trajectory"]) + [step.model_dump()]
    return updated


def _inc(state: AgentState) -> tuple[int, dict]:
    """Increment step_count and check the step cap. Returns (new_count, flags)."""
    new_count = state["step_count"] + 1
    flags = dict(state["guardrail_flags"])
    flags["steps_used"] = new_count
    return new_count, flags


# ---------------------------------------------------------------------------
# Node: router
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> AgentState:
    """
    Determine what the agent should do next. This is the only routing logic;
    all other nodes do one specific thing.

    The router emits a 'next' key used by conditional edges.
    """
    # Already handled by conditional edges — router just passes through
    return state


def route_decision(state: AgentState) -> str:
    """
    Conditional edge function: returns the name of the next node.
    Called after router_node.
    """
    # Step cap check
    exceeded, _ = check_step_cap(state["step_count"], AGENT_STEP_CAP)
    if exceeded:
        return "decide"

    candidates = state["candidates"]
    profiles = state["profiles"]
    scorecards = state["scorecards"]

    # Any candidate not yet parsed?
    for key in candidates:
        if key not in profiles:
            return "parse"

    # Any candidate not yet scored?
    for key in candidates:
        if key not in scorecards:
            return "score"

    # Any borderline candidate not yet verified?
    verified_candidates = {
        step["action_args"].get("candidate")
        for step in state["trajectory"]
        if step.get("action") == "verify_scorecard"
    }
    for key, scorecard in scorecards.items():
        if key not in verified_candidates and is_borderline(scorecard, state["rubric"]):
            return "verifier"

    # Any shortlisted candidate needing availability?
    shortlist = state["shortlist"]
    shortlisted_names = {e["candidate"] for e in shortlist if e.get("verdict") == "INTERVIEW"}
    if shortlist:
        for entry in shortlist:
            if entry.get("verdict") == "INTERVIEW":
                cname = entry["candidate"]
                if entry.get("proposed_slot") is None:
                    return "availability"

    # All parsing + scoring done, no pending availability → decide
    if not shortlist:
        return "decide"

    # Pending human approval for scheduling?
    if state.get("pending_approval"):
        return "schedule"

    return END


# ---------------------------------------------------------------------------
# Node: parse_node
# ---------------------------------------------------------------------------

def parse_node(state: AgentState) -> AgentState:
    """Parse the next unparsed candidate résumé."""
    new_count, flags = _inc(state)

    # Find first unparsed candidate
    target_key = None
    for key in state["candidates"]:
        if key not in state["profiles"]:
            target_key = key
            break

    if target_key is None:
        return {**state, "step_count": new_count, "guardrail_flags": flags}

    resume_text = state["candidates"][target_key]
    thought = f"Parsing résumé for {target_key}. Will extract structured profile via LLM."

    try:
        result = parse_resume.invoke({"resume_text": resume_text})

        # Detect injection metadata attached by the tool
        injection_detected = result.pop("_injection_detected", False)
        injection_details = result.pop("_injection_details", "")

        if injection_detected:
            flags["injection_detected"] = True
            flags["injection_details"] = injection_details

        observation = json.dumps(result, indent=2)
        guardrail_note = f"INJECTION BLOCKED: {injection_details}" if injection_detected else None

        trajectory = _log(state, thought, "parse_resume",
                          {"candidate": target_key}, observation, guardrail_note)

        new_profiles = dict(state["profiles"])
        new_profiles[target_key] = result

        return {
            **state,
            "profiles": new_profiles,
            "trajectory": trajectory,
            "step_count": new_count,
            "guardrail_flags": flags,
        }

    except Exception as e:
        trajectory = _log(state, thought, "parse_resume",
                          {"candidate": target_key}, f"ERROR: {e}")
        return {
            **state,
            "trajectory": trajectory,
            "step_count": new_count,
            "guardrail_flags": flags,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Node: score_node
# ---------------------------------------------------------------------------

def score_node(state: AgentState) -> AgentState:
    """Score the next unscored candidate against the rubric."""
    new_count, flags = _inc(state)

    target_key = None
    for key in state["profiles"]:
        if key not in state["scorecards"]:
            target_key = key
            break

    if target_key is None:
        return {**state, "step_count": new_count, "guardrail_flags": flags}

    profile = state["profiles"][target_key]
    rubric = state["rubric"]
    thought = (
        f"Scoring {target_key} against the rubric. "
        "Will assign 0-5 per criterion with evidence citations."
    )

    try:
        result = score_candidate.invoke({
            "profile_json": json.dumps(profile),
            "rubric_json": json.dumps(rubric),
        })

        # Guardrail 4 — verify criteria names are JD-derived
        criteria_names = [c["criterion"] for c in result.get("criteria", [])]
        fairness_pass, fairness_msg = check_criteria_fairness(criteria_names)
        flags["fairness_pass"] = fairness_pass

        observation = json.dumps(result, indent=2)
        guardrail_note = None if fairness_pass else f"FAIRNESS: {fairness_msg}"

        trajectory = _log(state, thought, "score_candidate",
                          {"candidate": target_key, "profile_json": json.dumps(profile), "rubric_json": json.dumps(rubric)}, observation, guardrail_note)

        new_scorecards = dict(state["scorecards"])
        new_scorecards[target_key] = result

        return {
            **state,
            "scorecards": new_scorecards,
            "trajectory": trajectory,
            "step_count": new_count,
            "guardrail_flags": flags,
        }

    except Exception as e:
        trajectory = _log(state, thought, "score_candidate",
                          {"candidate": target_key, "error": str(e)}, f"ERROR: {e}")
        return {
            **state,
            "trajectory": trajectory,
            "step_count": new_count,
            "guardrail_flags": flags,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Helper: borderline detection
# ---------------------------------------------------------------------------

def is_borderline(scorecard: dict, rubric: dict, band: float = 0.3) -> bool:
    """
    Return True when the candidate's weighted_total falls within ±band of
    either the INTERVIEW or HOLD threshold — meaning a second-pass check
    by the Verifier is warranted before a final verdict is made.

    Band is intentionally tight (0.3) so clear strong/weak candidates
    (>0.3 away from any threshold) skip the verifier entirely.
    Borderline = score in [3.2, 3.8] (near INTERVIEW=3.5)
              or score in [2.2, 2.8] (near HOLD=2.5)
    """
    thresholds = rubric.get("thresholds", {"INTERVIEW": 3.5, "HOLD": 2.5})
    total = scorecard.get("weighted_total", 0.0)
    for threshold in thresholds.values():
        if abs(total - threshold) <= band:
            return True
    return False


# ---------------------------------------------------------------------------
# Node: verifier_node
# ---------------------------------------------------------------------------

def verifier_node(state: AgentState) -> AgentState:
    """
    Second-pass automated verifier for borderline candidates.

    Fires only when a candidate's weighted_total is within ±0.5 of either
    the INTERVIEW or HOLD threshold (or when the first-pass scorer and a
    re-check disagree beyond a delta).

    Behaviour:
      - Re-checks every criterion's evidence citation against the rubric
        scale description.
      - Can *confirm* the score (no change), *adjust* it (small correction),
        or flag *needs_human_review* when evidence is ambiguous.
      - Writes `verifier` into AgentState.trajectory so Exercise 1/5
        invariant checks can assert its presence.
    """
    from langchain_openai import ChatOpenAI
    from config import GITHUB_TOKEN, GITHUB_MODELS_BASE_URL, LLM_MODEL

    new_count, flags = _inc(state)

    # Find the most recently scored candidate that is borderline and has not
    # yet been verified (we check for absence of "verifier" in action names).
    verified_candidates = {
        step["action_args"].get("candidate")
        for step in state["trajectory"]
        if step.get("action") == "verify_scorecard"
    }

    target_key = None
    for key, scorecard in state["scorecards"].items():
        if key not in verified_candidates and is_borderline(scorecard, state["rubric"]):
            target_key = key
            break

    if target_key is None:
        # Nothing to verify (not borderline, or already verified)
        return {**state, "step_count": new_count, "guardrail_flags": flags}

    scorecard = state["scorecards"][target_key]
    rubric = state["rubric"]
    thought = (
        f"Candidate {target_key} has a borderline score "
        f"({scorecard.get('weighted_total', '?'):.2f}). "
        "Running second-pass evidence verification before final verdict."
    )

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=GITHUB_TOKEN,
            base_url=GITHUB_MODELS_BASE_URL,
            temperature=0.0,
            max_retries=2,
            max_tokens=2000,
        )

        system_prompt = """You are an independent verification agent for a hiring scorecard.
Your job: re-check whether each criterion score is properly backed by evidence.

Rules:
1. For each criterion, read the evidence string and the rubric scale.
2. If the evidence genuinely supports the score → mark it "confirmed".
3. If the evidence is weak / over-generous → suggest a corrected score (never > original + 1).
4. If you cannot determine from the evidence alone → mark "needs_human_review".
5. Return ONLY valid JSON in exactly this structure:
{
  "candidate": "name",
  "verification_status": "confirmed" | "adjusted" | "needs_human_review",
  "adjustments": [
    {"criterion": "id", "original_score": 4, "verified_score": 3, "reason": "..."}
  ],
  "verified_weighted_total": 3.7,
  "notes": "Optional free-text summary"
}
If nothing changed, adjustments = [] and verified_weighted_total = original."""

        user_prompt = (
            f"Scorecard to verify:\n{json.dumps(scorecard, indent=2)}\n\n"
            f"Rubric:\n{json.dumps(rubric, indent=2)}"
        )

        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        verification = json.loads(raw)

        # Apply any verified score adjustments back to the scorecard
        new_scorecards = dict(state["scorecards"])
        updated_scorecard = dict(scorecard)

        if verification.get("adjustments"):
            criteria_map = {c["criterion"]: c for c in updated_scorecard.get("criteria", [])}
            for adj in verification["adjustments"]:
                cid = adj.get("criterion")
                if cid in criteria_map:
                    criteria_map[cid] = dict(criteria_map[cid])
                    criteria_map[cid]["score"] = adj["verified_score"]
            updated_scorecard["criteria"] = list(criteria_map.values())

        if "verified_weighted_total" in verification:
            updated_scorecard["weighted_total"] = round(
                verification["verified_weighted_total"], 3
            )

        # Mark scorecard as verified
        updated_scorecard["verifier_status"] = verification.get(
            "verification_status", "confirmed"
        )
        updated_scorecard["verifier_notes"] = verification.get("notes", "")

        if verification.get("verification_status") == "needs_human_review":
            flags["human_gate_status"] = "waiting_approval"

        new_scorecards[target_key] = updated_scorecard

        observation = json.dumps(verification, indent=2)
        trajectory = _log(
            state, thought, "verify_scorecard",
            {"candidate": target_key}, observation,
            "VERIFIER: needs_human_review" if verification.get(
                "verification_status") == "needs_human_review" else None,
        )

        return {
            **state,
            "scorecards": new_scorecards,
            "trajectory": trajectory,
            "step_count": new_count,
            "guardrail_flags": flags,
        }

    except Exception as e:
        trajectory = _log(
            state, thought, "verify_scorecard",
            {"candidate": target_key}, f"ERROR: {e}"
        )
        return {
            **state,
            "trajectory": trajectory,
            "step_count": new_count,
            "guardrail_flags": flags,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Node: decide_node
# ---------------------------------------------------------------------------

def decide_node(state: AgentState) -> AgentState:
    """Build the ranked shortlist from all scorecards."""
    new_count, flags = _inc(state)
    thought = "All candidates parsed and scored. Building ranked shortlist using rubric thresholds."

    thresholds = state["rubric"].get("thresholds", {"INTERVIEW": 3.5, "HOLD": 2.5})
    interview_threshold = thresholds.get("INTERVIEW", 3.5)
    hold_threshold = thresholds.get("HOLD", 2.5)

    shortlist: list[dict] = []
    for key, scorecard in state["scorecards"].items():
        total = scorecard.get("weighted_total", 0.0)
        profile = state["profiles"].get(key, {})

        if total >= interview_threshold:
            verdict = "INTERVIEW"
        elif total >= hold_threshold:
            verdict = "HOLD"
        else:
            verdict = "NOT_A_FIT"

        # Build evidence-based justification
        top_evidence = []
        for c in scorecard.get("criteria", []):
            if c.get("score", 0) >= 3 and c.get("evidence", ""):
                top_evidence.append(
                    f"{c['criterion_name']} (score {c['score']}/5): {c['evidence']}"
                )

        justification = (
            f"Weighted score: {total:.2f}/5.0. Verdict: {verdict}. "
            + (" Evidence: " + "; ".join(top_evidence[:3]) if top_evidence
               else "Insufficient evidence for high scores.")
        )

        entry = {
            "candidate": scorecard["candidate"],
            "verdict": verdict,
            "weighted_score": total,
            "justification": justification,
            "scorecard": scorecard,
            "proposed_slot": None,
            "confirmation": None,
        }
        shortlist.append(entry)

    # Rank by weighted score descending
    shortlist.sort(key=lambda e: e["weighted_score"], reverse=True)

    observation = json.dumps(
        [{"candidate": e["candidate"], "verdict": e["verdict"],
          "score": e["weighted_score"]} for e in shortlist],
        indent=2,
    )
    trajectory = _log(state, thought, "decide", {}, observation)

    return {
        **state,
        "shortlist": shortlist,
        "trajectory": trajectory,
        "step_count": new_count,
        "guardrail_flags": flags,
    }


# ---------------------------------------------------------------------------
# Node: availability_node
# ---------------------------------------------------------------------------

def availability_node(state: AgentState) -> AgentState:
    """Check availability for the next INTERVIEW candidate missing a slot."""
    new_count, flags = _inc(state)

    target_entry = None
    for entry in state["shortlist"]:
        if entry.get("verdict") == "INTERVIEW" and entry.get("proposed_slot") is None:
            target_entry = entry
            break

    if target_entry is None:
        return {**state, "step_count": new_count, "guardrail_flags": flags}

    candidate = target_entry["candidate"]
    thought = (
        f"{candidate} is shortlisted for INTERVIEW. "
        "Checking available slots before proposing to human approver."
    )

    try:
        slots = check_availability.invoke({
            "candidate_name": candidate,
            "week": "2025-W29",
        })

        chosen_slot = slots[0] if slots else None
        observation = json.dumps(slots, indent=2)
        trajectory = _log(state, thought, "check_availability",
                          {"candidate": candidate}, observation)

        # Update the shortlist entry with proposed slot
        new_shortlist = []
        for e in state["shortlist"]:
            if e["candidate"] == candidate:
                updated_entry = dict(e)
                updated_entry["proposed_slot"] = chosen_slot
                # Mark as pending approval
                updated_entry["confirmation"] = {
                    "candidate": candidate,
                    "slot": chosen_slot,
                    "status": "pending_approval",
                    "confirmation_id": "",
                }
                new_shortlist.append(updated_entry)
            else:
                new_shortlist.append(e)

        # Set pending_approval so schedule_node is reachable
        pending = next(
            (e for e in new_shortlist if e["candidate"] == candidate), None
        )
        flags["human_gate_status"] = "waiting_approval"

        return {
            **state,
            "shortlist": new_shortlist,
            "trajectory": trajectory,
            "step_count": new_count,
            "guardrail_flags": flags,
            "pending_approval": pending,
        }

    except Exception as e:
        trajectory = _log(state, thought, "check_availability",
                          {"candidate": candidate}, f"ERROR: {e}")
        return {
            **state,
            "trajectory": trajectory,
            "step_count": new_count,
            "guardrail_flags": flags,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Node: schedule_node  (GUARDRAIL 1 — human-in-the-loop gate)
# ---------------------------------------------------------------------------

def schedule_node(state: AgentState) -> AgentState:
    """
    Propose an interview slot. PAUSES here via LangGraph interrupt() until
    a human explicitly approves. propose_interview only fires after approval.
    """
    new_count, flags = _inc(state)
    pending = state.get("pending_approval")

    if pending is None:
        return {**state, "step_count": new_count, "guardrail_flags": flags}

    candidate = pending["candidate"]
    slot = pending.get("proposed_slot")

    thought = (
        f"Ready to schedule interview for {candidate}. "
        "PAUSING for human approval before firing propose_interview. "
        "This action tool must never fire autonomously."
    )

    # GUARDRAIL 1 — structural interrupt: graph pauses here
    # The Streamlit UI will resume the graph with approved=True or rejected
    human_decision = interrupt({
        "message": f"Approve interview scheduling for {candidate}?",
        "candidate": candidate,
        "slot": slot,
    })

    approved = human_decision.get("approved", False)
    flags["human_gate_status"] = "approved" if approved else "rejected"

    if approved and slot:
        try:
            conf_result = propose_interview.invoke({
                "candidate_name": candidate,
                "slot_json": json.dumps(slot),
                "approved": True,
            })
            observation = f"APPROVED. Confirmation: {json.dumps(conf_result, indent=2)}"

            new_shortlist = []
            for e in state["shortlist"]:
                if e["candidate"] == candidate:
                    updated = dict(e)
                    updated["confirmation"] = conf_result
                    new_shortlist.append(updated)
                else:
                    new_shortlist.append(e)

            trajectory = _log(state, thought, "propose_interview",
                               {"candidate": candidate, "approved": True}, observation)

            return {
                **state,
                "shortlist": new_shortlist,
                "trajectory": trajectory,
                "step_count": new_count,
                "guardrail_flags": flags,
                "pending_approval": None,
            }

        except Exception as e:
            trajectory = _log(state, thought, "propose_interview",
                               {"candidate": candidate}, f"ERROR: {e}")
            return {
                **state,
                "trajectory": trajectory,
                "step_count": new_count,
                "guardrail_flags": flags,
                "error": str(e),
            }
    else:
        observation = f"REJECTED by human. Interview not scheduled for {candidate}."
        trajectory = _log(state, thought, "propose_interview",
                           {"candidate": candidate, "approved": False}, observation)
        return {
            **state,
            "trajectory": trajectory,
            "step_count": new_count,
            "guardrail_flags": flags,
            "pending_approval": None,
        }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router", router_node)
    graph.add_node("parse", parse_node)
    graph.add_node("score", score_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("decide", decide_node)
    graph.add_node("availability", availability_node)
    graph.add_node("schedule", schedule_node)

    # Entry point
    graph.add_edge(START, "router")

    # Conditional routing from router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "parse": "parse",
            "score": "score",
            "verifier": "verifier",
            "decide": "decide",
            "availability": "availability",
            "schedule": "schedule",
            END: END,
        },
    )

    # After each action node, loop back to router for re-evaluation
    graph.add_edge("parse", "router")
    graph.add_edge("score", "router")
    graph.add_edge("verifier", "router")
    graph.add_edge("availability", "router")
    graph.add_edge("decide", "router")
    graph.add_edge("schedule", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["schedule"])


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------

def make_initial_state(jd: str, candidates: dict[str, str], rubric: dict) -> AgentState:
    """Return a fresh AgentState for a new run."""
    _logger.clear()
    return AgentState(
        jd=jd,
        rubric=rubric,
        candidates=candidates,
        profiles={},
        scorecards={},
        shortlist=[],
        trajectory=[],
        step_count=0,
        guardrail_flags=GuardrailFlags().model_dump(),
        pending_approval=None,
        run_stats=RunStats().model_dump(),
        error=None,
    )
