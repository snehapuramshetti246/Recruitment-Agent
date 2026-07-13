"""
eval/crew_wrapper.py — Thin adapter exposing run_crew(input) for all eval tools.

This is the single point of contact between the eval suite and the LangGraph
agent. It wraps agent.py's build_graph() + make_initial_state() and returns
a standardised result dict that every eval file can consume.

The wrapper:
  - Builds a fresh graph on each call (isolated MemorySaver per run).
  - Converts the AgentState TypedDict into a plain dict for portability.
  - Does NOT auto-approve the human gate — automated eval runs will
    halt at schedule_node (the interrupt), which is the expected behaviour.
    test_gate.py asserts on this halt explicitly.
  - Exposes crew_predict(df) for Giskard / Promptfoo integration.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import build_graph, make_initial_state, is_borderline
from config import load_jd, RUBRIC


# ---------------------------------------------------------------------------
# Primary interface
# ---------------------------------------------------------------------------

def run_crew(input_dict: dict) -> dict:
    """
    Run the recruitment agent for a single eval task.

    Args:
        input_dict: Must contain:
            - 'jd'         (str)  — job description text
            - 'rubric'     (dict) — scoring rubric
            - 'candidates' (dict) — {key: resume_text}

          Optional:
            - '_mock_availability_override' (list) — override for check_availability
              results (used in T10 conflicting-availability task). Empty list = no slots.

    Returns:
        dict with keys:
            - 'trajectory'      list[dict]  — full thought/action/observation log
            - 'shortlist'       list[dict]  — ShortlistEntry dicts
            - 'scorecards'      dict        — {candidate_key: ScoreCard dict}
            - 'profiles'        dict        — {candidate_key: CandidateProfile dict}
            - 'guardrail_flags' dict        — GuardrailFlags dict
            - 'step_count'      int
            - 'pending_approval' dict|None  — if gate halted before schedule
            - 'error'           str|None
    """
    jd = input_dict.get("jd", load_jd())
    rubric = input_dict.get("rubric", RUBRIC)
    candidates = input_dict.get("candidates", {})

    if not candidates:
        return {
            "trajectory": [],
            "shortlist": [],
            "scorecards": {},
            "profiles": {},
            "guardrail_flags": {},
            "step_count": 0,
            "pending_approval": None,
            "error": "No candidates provided",
        }

    # Handle mock availability override for T10
    avail_override = input_dict.get("_mock_availability_override")
    if avail_override is not None:
        _patch_availability(avail_override)

    # Build fresh graph and initial state
    graph = build_graph()
    state = make_initial_state(jd=jd, candidates=candidates, rubric=rubric)

    # Generate a unique thread ID so each eval run has its own checkpoint
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    try:
        # Run the graph — it will halt at schedule_node (interrupt_before=['schedule'])
        # if any candidate reaches INTERVIEW. That halt is the expected behaviour.
        result_state = graph.invoke(state, config=config)
    except Exception as e:
        # Re-try once (transient LLM errors happen)
        try:
            result_state = graph.invoke(state, config=config)
        except Exception as e2:
            return {
                "trajectory": [],
                "shortlist": [],
                "scorecards": {},
                "profiles": {},
                "guardrail_flags": {},
                "step_count": 0,
                "pending_approval": None,
                "error": f"Agent run failed: {e2}",
            }

    # Restore any availability patch
    if avail_override is not None:
        _restore_availability()

    # Normalise — if graph.invoke returns None or stops at an interrupt,
    # fetch the current state snapshot instead.
    if result_state is None:
        snap = graph.get_state(config)
        result_state = snap.values if snap else {}

    return {
        "trajectory": result_state.get("trajectory", []),
        "shortlist": result_state.get("shortlist", []),
        "scorecards": result_state.get("scorecards", {}),
        "profiles": result_state.get("profiles", {}),
        "guardrail_flags": result_state.get("guardrail_flags", {}),
        "step_count": result_state.get("step_count", 0),
        "pending_approval": result_state.get("pending_approval"),
        "error": result_state.get("error"),
    }


# ---------------------------------------------------------------------------
# Giskard / Promptfoo interface
# ---------------------------------------------------------------------------

def build_input(question: str) -> dict:
    """
    Convert a free-text question (used by red-team tools) into an agent
    input dict. Treats the question as a candidate résumé and uses the
    default JD + rubric.
    """
    return {
        "jd": load_jd(),
        "rubric": RUBRIC,
        "candidates": {"red_team_candidate": question},
    }


def crew_predict(df: Any) -> list[str]:
    """
    Giskard/Promptfoo-compatible prediction function.

    Args:
        df: A pandas DataFrame with an 'input' column (or any iterable
            of dicts with an 'input' key).

    Returns:
        List of summary strings, one per input row.
    """
    try:
        # pandas DataFrame
        rows = df["input"].tolist()
    except (TypeError, KeyError, AttributeError):
        # Fallback: treat as plain list
        rows = list(df) if hasattr(df, "__iter__") else [str(df)]

    results = []
    for row in rows:
        try:
            input_dict = build_input(str(row))
            run_result = run_crew(input_dict)
            shortlist = run_result.get("shortlist", [])
            if shortlist:
                summary_parts = []
                for entry in shortlist:
                    summary_parts.append(
                        f"{entry.get('candidate', '?')}: "
                        f"{entry.get('verdict', '?')} "
                        f"(score={entry.get('weighted_score', 0):.2f})"
                    )
                summary = " | ".join(summary_parts)
            else:
                error = run_result.get("error")
                summary = f"No decision produced. Error: {error}" if error else "No decision produced."
            results.append(summary)
        except Exception as e:  # noqa: BLE001
            results.append(f"ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
# Mock availability patch helper (for T10)
# ---------------------------------------------------------------------------

_original_check_availability_func = None


def _patch_availability(override_slots: list) -> None:
    """
    Monkey-patch check_availability to return override_slots.
    Used only for T10 (conflicting/empty availability task).
    """
    global _original_check_availability_func
    import tools as tools_module

    _original_check_availability_func = tools_module.check_availability

    # Create a replacement tool that returns the override
    from langchain_core.tools import tool as langtool

    @langtool
    def check_availability_override(candidate_name: str, week: str) -> list[dict]:  # type: ignore[override]
        """Mock: return override slots."""
        return override_slots

    tools_module.check_availability = check_availability_override


def _restore_availability() -> None:
    """Restore the original check_availability after a patched run."""
    global _original_check_availability_func
    if _original_check_availability_func is not None:
        import tools as tools_module
        tools_module.check_availability = _original_check_availability_func
        _original_check_availability_func = None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running crew_wrapper smoke test…")
    from config import load_resumes
    resumes = load_resumes()
    first_key = next(iter(resumes))
    test_input = {
        "jd": load_jd(),
        "rubric": RUBRIC,
        "candidates": {first_key: resumes[first_key]},
    }
    result = run_crew(test_input)
    print(f"Shortlist: {json.dumps(result['shortlist'], indent=2, default=str)[:400]}…")
    print(f"Steps: {result['step_count']}, Error: {result['error']}")
