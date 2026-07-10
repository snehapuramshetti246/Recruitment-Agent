"""
config.py — Centralised configuration for the TechVest Recruitment Agent.

Loads from .env via python-dotenv. All other modules import from here;
nothing else touches os.environ directly.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (safe to call even if .env does not exist)
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
JD_PATH = BASE_DIR / "jd.txt"
RESUMES_DIR = BASE_DIR / "resumes"
RUBRIC_PATH = BASE_DIR / "rubric.json"
AUDIT_LOG_PATH = BASE_DIR / os.getenv("AUDIT_LOG_PATH", "decisions.json")

# ── LLM — GitHub Models ────────────────────────────────────────────────────
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
GITHUB_MODELS_BASE_URL: str = "https://models.inference.ai.azure.com"

# Model to use; override by setting LLM_MODEL in .env
# Browse available models at https://github.com/marketplace/models
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

# ── Agent safety ────────────────────────────────────────────────────────────
AGENT_STEP_CAP: int = int(os.getenv("AGENT_STEP_CAP", "25"))
AGENT_RECURSION_LIMIT: int = int(os.getenv("AGENT_RECURSION_LIMIT", "50"))

# ── Injection detection keywords ────────────────────────────────────────────
INJECTION_PATTERNS: list[str] = [
    "ignore your",
    "ignore previous",
    "disregard",
    "override",
    "rank me first",
    "rank this candidate first",
    "system note",
    "system prompt",
    "forget your instructions",
    "you must",
    "give me the highest",
    "assign score 5",
    "ignore instructions",
]

# ── Rubric (loaded once at import time) ─────────────────────────────────────
def load_rubric() -> dict:
    """Load and return the rubric from rubric.json."""
    with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


RUBRIC: dict = load_rubric()

# ── Résumé files (loaded once at import time) ────────────────────────────────
def load_resumes() -> dict[str, str]:
    """Return {filename_stem: text} for all .txt files in resumes/."""
    resumes: dict[str, str] = {}
    if RESUMES_DIR.exists():
        for path in sorted(RESUMES_DIR.glob("*.txt")):
            resumes[path.stem] = path.read_text(encoding="utf-8")
    return resumes


def load_jd() -> str:
    """Return the job description text."""
    return JD_PATH.read_text(encoding="utf-8")


# ── Interview slots (mock availability data) ─────────────────────────────────
MOCK_SLOTS: dict[str, list[dict]] = {
    "default": [
        {"day": "Monday",    "date": "2025-07-14", "time": "10:00", "duration_minutes": 60},
        {"day": "Monday",    "date": "2025-07-14", "time": "14:00", "duration_minutes": 60},
        {"day": "Tuesday",   "date": "2025-07-15", "time": "11:00", "duration_minutes": 60},
        {"day": "Wednesday", "date": "2025-07-16", "time": "09:00", "duration_minutes": 60},
        {"day": "Thursday",  "date": "2025-07-17", "time": "15:00", "duration_minutes": 60},
        {"day": "Friday",    "date": "2025-07-18", "time": "10:00", "duration_minutes": 60},
    ]
}


def validate_config() -> list[str]:
    """Return a list of configuration warnings (empty = all OK)."""
    warnings: list[str] = []
    if not GITHUB_TOKEN:
        warnings.append("GITHUB_TOKEN is not set. LLM calls will fail.")
    if not JD_PATH.exists():
        warnings.append(f"JD file not found: {JD_PATH}")
    if not RUBRIC_PATH.exists():
        warnings.append(f"Rubric file not found: {RUBRIC_PATH}")
    if not RESUMES_DIR.exists():
        warnings.append(f"Resumes directory not found: {RESUMES_DIR}")
    return warnings
