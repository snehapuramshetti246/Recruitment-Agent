"""
audit.py — Trajectory logging and decision audit log for the TechVest Recruitment Agent.

Guardrail 5: Every run is persisted to decisions.json with full trajectory,
             shortlist, and guardrail status so any decision can be reconstructed.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import AUDIT_LOG_PATH
from models import TrajectoryStep, ShortlistEntry, GuardrailFlags, RunStats


# ---------------------------------------------------------------------------
# In-memory trajectory accumulator
# ---------------------------------------------------------------------------

class TrajectoryLogger:
    """
    Accumulates thought/action/observation steps during an agent run.
    Flushed to AuditLog at the end of the run.
    """

    def __init__(self) -> None:
        self._steps: list[TrajectoryStep] = []
        self._step_counter: int = 0

    def log(
        self,
        thought: str,
        action: str,
        action_args: dict | None = None,
        observation: str = "",
        guardrail_triggered: str | None = None,
    ) -> TrajectoryStep:
        """Append a step and return it."""
        self._step_counter += 1
        step = TrajectoryStep(
            step_number=self._step_counter,
            thought=thought,
            action=action,
            action_args=action_args or {},
            observation=observation,
            guardrail_triggered=guardrail_triggered,
        )
        self._steps.append(step)
        return step

    def steps(self) -> list[TrajectoryStep]:
        return list(self._steps)

    def step_count(self) -> int:
        return self._step_counter

    def clear(self) -> None:
        self._steps = []
        self._step_counter = 0


# ---------------------------------------------------------------------------
# Audit log — persists runs to decisions.json
# ---------------------------------------------------------------------------

class AuditLog:
    """
    Append-only JSON audit log. Each entry is one complete agent run.

    File format: a JSON array of run records.
    We load the array, append, and write back on each save.
    """

    def __init__(self, path: Path = AUDIT_LOG_PATH) -> None:
        self.path = Path(path)

    # -- Reading ---------------------------------------------------------------

    def load_all(self) -> list[dict]:
        """Return all stored run records."""
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def get_run(self, run_id: str) -> dict | None:
        """Return a single run record by ID, or None if not found."""
        for record in self.load_all():
            if record.get("run_id") == run_id:
                return record
        return None

    # -- Writing ---------------------------------------------------------------

    def save_run(
        self,
        trajectory: list[TrajectoryStep],
        shortlist: list[ShortlistEntry],
        guardrail_flags: GuardrailFlags,
        run_stats: RunStats,
        run_id: str | None = None,
    ) -> str:
        """
        Persist a completed run. Returns the run_id.
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        record = _build_record(
            run_id=run_id,
            trajectory=trajectory,
            shortlist=shortlist,
            guardrail_flags=guardrail_flags,
            run_stats=run_stats,
        )

        existing = self.load_all()
        existing.append(record)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, default=str)

        return run_id

    def update_confirmation(self, run_id: str, candidate: str, status: str) -> bool:
        """
        Update the confirmation status for a candidate in a saved run.
        Returns True if updated, False if run_id not found.
        """
        records = self.load_all()
        updated = False
        for record in records:
            if record.get("run_id") == run_id:
                for entry in record.get("shortlist", []):
                    if entry.get("candidate") == candidate:
                        if entry.get("confirmation"):
                            entry["confirmation"]["status"] = status
                        updated = True
        if updated:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False, default=str)
        return updated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_record(
    run_id: str,
    trajectory: list[TrajectoryStep],
    shortlist: list[ShortlistEntry],
    guardrail_flags: GuardrailFlags,
    run_stats: RunStats,
) -> dict:
    """Serialise a run to a plain dict for JSON storage."""
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_stats": run_stats.model_dump(),
        "guardrail_flags": guardrail_flags.model_dump(),
        "shortlist": [_serialise_shortlist_entry(e) for e in shortlist],
        "trajectory": [_serialise_step(s) for s in trajectory],
    }


def _serialise_step(step: TrajectoryStep) -> dict:
    return {
        "step_number": step.step_number,
        "thought": step.thought,
        "action": step.action,
        "action_args": step.action_args,
        "observation": step.observation,
        "guardrail_triggered": step.guardrail_triggered,
    }


def _serialise_shortlist_entry(entry: ShortlistEntry) -> dict:
    d = entry.model_dump()
    return d


# ---------------------------------------------------------------------------
# Convenience: pretty-print trajectory to string (for UI)
# ---------------------------------------------------------------------------

def format_trajectory_text(steps: list[TrajectoryStep]) -> str:
    """Return a human-readable text representation of the trajectory."""
    lines: list[str] = []
    for step in steps:
        lines.append(f"\n{'='*60}")
        lines.append(f"Step {step.step_number}")
        lines.append(f"{'='*60}")
        lines.append(f"THOUGHT: {step.thought}")
        lines.append(f"ACTION:  {step.action}({json.dumps(step.action_args, indent=2)})")
        lines.append(f"OBSERVATION:\n{step.observation}")
        if step.guardrail_triggered:
            lines.append(f"⚠️  GUARDRAIL: {step.guardrail_triggered}")
    return "\n".join(lines)
