"""
guardrails.py — All 5 required guardrails for the TechVest Recruitment Agent.

Guardrail 1: Human-in-the-loop gate (enforced structurally via LangGraph interrupt)
Guardrail 2: Step / iteration cap
Guardrail 3: Prompt-injection defence
Guardrail 4: Fairness check (JD-only criteria, name-swap test)
Guardrail 5: Decision audit log (see audit.py)
"""

from __future__ import annotations

import re
from typing import Optional

from config import INJECTION_PATTERNS, AGENT_STEP_CAP
from models import GuardrailFlags, ScoreCard, CandidateProfile


# ---------------------------------------------------------------------------
# Guardrail 3 — Prompt-injection defence
# ---------------------------------------------------------------------------

def scan_for_injection(text: str) -> tuple[bool, str]:
    """
    Scan résumé text for prompt-injection patterns.

    Returns:
        (injection_detected: bool, details: str)
    """
    lower_text = text.lower()
    triggered: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.lower() in lower_text:
            # Find the context around the match
            idx = lower_text.find(pattern.lower())
            start = max(0, idx - 30)
            end = min(len(text), idx + len(pattern) + 60)
            context_snippet = text[start:end].replace("\n", " ").strip()
            triggered.append(f'Pattern "{pattern}" found near: "…{context_snippet}…"')

    if triggered:
        details = " | ".join(triggered)
        return True, details
    return False, ""


def sanitise_resume(text: str) -> tuple[str, bool, str]:
    """
    Remove or neutralise injection lines from résumé text before sending to LLM.

    Returns:
        (sanitised_text, injection_detected, injection_details)
    """
    detected, details = scan_for_injection(text)
    if not detected:
        return text, False, ""

    lines = text.split("\n")
    clean_lines: list[str] = []
    removed_lines: list[str] = []

    lower_patterns = [p.lower() for p in INJECTION_PATTERNS]

    for line in lines:
        line_lower = line.lower()
        is_injected = any(p in line_lower for p in lower_patterns)
        if is_injected:
            removed_lines.append(line.strip())
            # Replace with a safe marker that is transparent in the trajectory
            clean_lines.append("[LINE REMOVED BY INJECTION GUARDRAIL]")
        else:
            clean_lines.append(line)

    sanitised = "\n".join(clean_lines)
    full_details = (
        f"INJECTION DETECTED AND NEUTRALISED. "
        f"Removed {len(removed_lines)} line(s): {removed_lines}. "
        f"Detection context: {details}"
    )
    return sanitised, True, full_details


# ---------------------------------------------------------------------------
# Guardrail 2 — Step cap
# ---------------------------------------------------------------------------

def check_step_cap(steps_used: int, cap: int = AGENT_STEP_CAP) -> tuple[bool, str]:
    """
    Returns (exceeded: bool, message: str).
    Call before each agent step.
    """
    if steps_used >= cap:
        return True, (
            f"STEP CAP REACHED: {steps_used}/{cap} steps used. "
            "Agent halted to prevent runaway loop."
        )
    return False, f"{steps_used}/{cap} steps used"


# ---------------------------------------------------------------------------
# Guardrail 4 — Fairness check
# ---------------------------------------------------------------------------

PROTECTED_TERMS: list[str] = [
    # Names / identity
    "name", "first name", "surname", "gender", "sex", "age", "religion",
    "caste", "nationality", "ethnicity", "race",
    # College prestige (explicitly excluded by JD)
    "iit", "iim", "nit", "bits", "college prestige", "college ranking",
    "gpa", "cgpa", "grades", "marks", "percentage",
    # Irrelevant experience
    "years of employment", "formal employment",
]


def check_criteria_fairness(criteria_used: list[str]) -> tuple[bool, str]:
    """
    Verify that none of the scoring criteria touch protected attributes.
    Returns (pass: bool, details: str).
    """
    violations: list[str] = []
    for criterion in criteria_used:
        c_lower = criterion.lower()
        for term in PROTECTED_TERMS:
            if term in c_lower:
                violations.append(f"Criterion '{criterion}' contains protected term '{term}'")

    if violations:
        return False, "FAIRNESS VIOLATION: " + "; ".join(violations)
    return True, "All criteria are JD-derived and do not reference protected attributes."


def name_swap_test(
    scorecard_a: ScoreCard,
    scorecard_b: ScoreCard,
    name_a: str,
    name_b: str,
) -> tuple[bool, str]:
    """
    Verify that two scorecards (same profile, different names) produce the same weighted total.
    Returns (pass: bool, report: str).
    """
    diff = abs(scorecard_a.weighted_total - scorecard_b.weighted_total)
    passed = diff < 0.01  # allow floating-point epsilon

    criterion_diffs: list[str] = []
    for ca, cb in zip(scorecard_a.criteria, scorecard_b.criteria):
        if ca.score != cb.score:
            criterion_diffs.append(
                f"  Criterion '{ca.criterion}': {name_a}={ca.score}, {name_b}={cb.score}"
            )

    if passed:
        report = (
            f"NAME-SWAP TEST PASSED ✅\n"
            f"  {name_a}: {scorecard_a.weighted_total:.2f}\n"
            f"  {name_b}: {scorecard_b.weighted_total:.2f}\n"
            f"  Δ = {diff:.4f} (within tolerance)"
        )
    else:
        report = (
            f"NAME-SWAP TEST FAILED ❌\n"
            f"  {name_a}: {scorecard_a.weighted_total:.2f}\n"
            f"  {name_b}: {scorecard_b.weighted_total:.2f}\n"
            f"  Δ = {diff:.4f}\n"
            f"  Per-criterion differences:\n" + "\n".join(criterion_diffs)
        )
    return passed, report


# ---------------------------------------------------------------------------
# Guardrail 1 — Human-gate enforcement helper
# ---------------------------------------------------------------------------

def assert_human_approved(confirmation_status: str, candidate: str) -> None:
    """
    Raise a RuntimeError if propose_interview is called without approval.
    This is a programmatic last-resort safety net; the primary enforcement
    is the LangGraph interrupt() in the schedule_node.
    """
    if confirmation_status != "confirmed":
        raise RuntimeError(
            f"SAFETY VIOLATION: propose_interview called for '{candidate}' "
            f"without human approval (status='{confirmation_status}'). "
            "Blocked by human-gate guardrail."
        )


# ---------------------------------------------------------------------------
# Guardrail status summary (for UI display)
# ---------------------------------------------------------------------------

def get_guardrail_summary(flags: GuardrailFlags) -> dict:
    """Return a dict suitable for rendering in the Streamlit sidebar."""
    step_exceeded, step_msg = check_step_cap(flags.steps_used, flags.step_cap)
    return {
        "step_cap": {
            "label": "Step Cap",
            "ok": not step_exceeded,
            "detail": step_msg,
        },
        "human_gate": {
            "label": "Human-in-the-Loop Gate",
            "ok": flags.human_gate_status in ("armed", "approved"),
            "detail": {
                "armed": "Armed — no interview scheduled yet",
                "waiting_approval": "⏳ Waiting for human approval",
                "approved": "✅ Approved by human",
                "rejected": "🚫 Rejected by human",
            }.get(flags.human_gate_status, flags.human_gate_status),
        },
        "injection": {
            "label": "Prompt-Injection Defence",
            "ok": not flags.injection_detected,
            "detail": (
                f"⚠️ Blocked 1 attempt — {flags.injection_details}"
                if flags.injection_detected
                else "No injection detected"
            ),
        },
        "fairness": {
            "label": "Fairness Check",
            "ok": flags.fairness_pass is True,
            "detail": (
                "✅ Pass" if flags.fairness_pass is True
                else "❌ Fail" if flags.fairness_pass is False
                else "Not yet tested"
            ),
        },
        "audit_log": {
            "label": "Decision Audit Log",
            "ok": True,
            "detail": f"Logging to {flags.audit_log_path}",
        },
    }
