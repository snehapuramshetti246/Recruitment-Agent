"""
eval/test_output.py — Exercise 3: Output quality using DeepEval.

Metrics:
  - FaithfulnessMetric (threshold ≥ 0.8)
      → justification only cites evidence that actually exists in the résumé
  - AnswerRelevancyMetric (threshold ≥ 0.7)
      → justification is relevant to the JD requirement
  - Task-completion check (every candidate has a decision + cited evidence)
  - Fairness / name-swap check (swapping candidate name must not change ranking)

Run:
    python -m pytest eval/test_output.py -v
    # or directly:
    python eval/test_output.py
"""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.tasks import ALL_TASKS, EvalTask, STRONG_FIT_TASK_IDS
from eval.crew_wrapper import run_crew

# ---------------------------------------------------------------------------
# DeepEval imports — graceful fallback if unavailable
# ---------------------------------------------------------------------------

try:
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    LLMTestCase = None  # type: ignore[assignment,misc]
    FaithfulnessMetric = None  # type: ignore[assignment,misc]
    AnswerRelevancyMetric = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# GitHub Models judge for DeepEval (avoids needing OPENAI_API_KEY)
# ---------------------------------------------------------------------------

def _make_deepeval_model():
    """
    Returns a DeepEvalBaseLLM instance backed by the project's GitHub Models
    credentials so DeepEval metrics don't require OPENAI_API_KEY.
    Returns None if deepeval is not available.
    """
    if not DEEPEVAL_AVAILABLE:
        return None
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
        from langchain_openai import ChatOpenAI
        from config import GITHUB_TOKEN, GITHUB_MODELS_BASE_URL, LLM_MODEL

        class GitHubModelsJudge(DeepEvalBaseLLM):
            def __init__(self):
                self._token    = GITHUB_TOKEN
                self._base_url = GITHUB_MODELS_BASE_URL
                self._model    = LLM_MODEL

            def load_model(self):
                return ChatOpenAI(
                    model=self._model,
                    api_key=self._token,
                    base_url=self._base_url,
                    temperature=0.0,
                    max_retries=2,
                    max_tokens=1024,
                )

            def generate(self, prompt: str) -> str:
                return self.load_model().invoke(prompt).content

            async def a_generate(self, prompt: str) -> str:
                return self.generate(prompt)

            def get_model_name(self) -> str:
                return self._model

        return GitHubModelsJudge()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper: extract shortlist entries from a run result
# ---------------------------------------------------------------------------

def extract_shortlist_entries(run_result: dict) -> list[dict]:
    """Return the agent's shortlist (list of decision dicts)."""
    return run_result.get("shortlist", [])


def extract_decision_for_candidate(run_result: dict, candidate_key: str) -> dict | None:
    """Find the shortlist entry for a specific candidate key."""
    for entry in run_result.get("shortlist", []):
        cname = entry.get("candidate", "")
        if candidate_key.lower() in cname.lower() or cname.lower() in candidate_key.lower():
            return entry
    return None


# ---------------------------------------------------------------------------
# Task-completion check (deterministic, no LLM)
# ---------------------------------------------------------------------------

def task_completion_check(run_result: dict, task: EvalTask) -> tuple[bool, list[str]]:
    """
    Assert:
      1. A shortlist entry exists for every candidate in the task.
      2. Each entry has a non-empty verdict.
      3. Each entry has a non-empty justification.
      4. Each entry with score > 0 has cited evidence.
    """
    issues: list[str] = []
    shortlist = extract_shortlist_entries(run_result)
    candidate_keys = list(task.input.get("candidates", {}).keys())

    # 1. Every candidate has a result
    found_names = {e.get("candidate", "").lower() for e in shortlist}
    for key in candidate_keys:
        name_found = any(
            key.lower() in fname or fname in key.lower()
            for fname in found_names
        )
        if not name_found:
            issues.append(f"COMPLETION-1: no shortlist entry for candidate key {key!r}")

    # 2–4. Quality of each entry
    for entry in shortlist:
        cname = entry.get("candidate", "?")
        verdict = entry.get("verdict", "")
        justification = entry.get("justification", "")
        score = entry.get("weighted_score", 0.0)

        if not verdict:
            issues.append(f"COMPLETION-2: {cname} has no verdict")

        if not justification:
            issues.append(f"COMPLETION-3: {cname} has no justification")

        if score > 0 and not justification:
            issues.append(f"COMPLETION-4: {cname} score={score:.2f} but no evidence cited")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Faithfulness check via DeepEval
# ---------------------------------------------------------------------------

def faithfulness_check(
    run_result: dict,
    task: EvalTask,
    threshold: float = 0.8,
) -> dict:
    """
    For each candidate in the task, run FaithfulnessMetric.

    'Faithfulness' = the justification only cites facts that exist in
    the résumé text that was actually provided (retrieval_context).
    """
    results = {}

    if not DEEPEVAL_AVAILABLE:
        return {
            "skipped": True,
            "reason": "deepeval not installed",
            "threshold": threshold,
        }

    candidate_texts = task.input.get("candidates", {})
    shortlist = extract_shortlist_entries(run_result)

    for entry in shortlist:
        cname = entry.get("candidate", "")
        justification = entry.get("justification", "")

        # Find the résumé text for this candidate
        resume_text = ""
        for key, text in candidate_texts.items():
            if key.lower() in cname.lower() or cname.lower() in key.lower():
                resume_text = text
                break

        if not resume_text or not justification:
            results[cname] = {
                "score": None,
                "pass": False,
                "reason": "Missing justification or résumé text",
            }
            continue

        try:
            jd_requirement = task.input.get("jd", "")[:200]
            tc = LLMTestCase(
                input=jd_requirement,
                actual_output=justification,
                retrieval_context=[resume_text],
            )
            judge = _make_deepeval_model()
            metric = FaithfulnessMetric(
                threshold=threshold,
                model=judge,
                verbose_mode=False,
            )
            metric.measure(tc)
            results[cname] = {
                "score": metric.score,
                "pass": metric.score >= threshold if metric.score is not None else False,
                "reason": metric.reason,
                "threshold": threshold,
            }
        except Exception as e:  # noqa: BLE001
            results[cname] = {
                "score": None,
                "pass": False,
                "reason": f"Metric error: {e}",
            }

    return results


# ---------------------------------------------------------------------------
# Answer relevancy check via DeepEval
# ---------------------------------------------------------------------------

def relevancy_check(
    run_result: dict,
    task: EvalTask,
    threshold: float = 0.7,
) -> dict:
    """
    For each candidate in the task, run AnswerRelevancyMetric.

    'Relevancy' = the justification is relevant to what the JD is asking.
    """
    results = {}

    if not DEEPEVAL_AVAILABLE:
        return {
            "skipped": True,
            "reason": "deepeval not installed",
            "threshold": threshold,
        }

    shortlist = extract_shortlist_entries(run_result)
    jd_requirement = task.input.get("jd", "")[:400]

    for entry in shortlist:
        cname = entry.get("candidate", "")
        justification = entry.get("justification", "")

        if not justification:
            results[cname] = {
                "score": None,
                "pass": False,
                "reason": "No justification to evaluate",
            }
            continue

        try:
            tc = LLMTestCase(
                input=jd_requirement,
                actual_output=justification,
            )
            judge = _make_deepeval_model()
            metric = AnswerRelevancyMetric(
                threshold=threshold,
                model=judge,
                verbose_mode=False,
            )
            metric.measure(tc)
            results[cname] = {
                "score": metric.score,
                "pass": metric.score >= threshold if metric.score is not None else False,
                "reason": metric.reason,
                "threshold": threshold,
            }
        except Exception as e:  # noqa: BLE001
            results[cname] = {
                "score": None,
                "pass": False,
                "reason": f"Metric error: {e}",
            }

    return results


# ---------------------------------------------------------------------------
# Fairness / name-swap check
# ---------------------------------------------------------------------------

def swap_candidate_name(input_dict: dict, old_name: str, new_name: str) -> dict:
    """
    Return a deep copy of input_dict with the candidate key renamed and
    the résumé text's first occurrence of the old name replaced.
    """
    swapped = copy.deepcopy(input_dict)
    candidates = swapped.get("candidates", {})

    # Rename the key
    if old_name in candidates:
        text = candidates.pop(old_name)
        # Replace name in résumé text (first occurrence, case-insensitive)
        # Build a display name from the key: "priya_nair" → "Priya Nair"
        display_old = old_name.replace("_", " ").title()
        display_new = new_name.replace("_", " ").title()
        text = text.replace(display_old, display_new, 1)
        candidates[new_name] = text

    swapped["candidates"] = candidates
    return swapped


def _get_verdict(run_result: dict, candidate_key: str) -> str | None:
    """Extract the verdict for a candidate from a run result."""
    for entry in run_result.get("shortlist", []):
        cname = entry.get("candidate", "").lower()
        if candidate_key.lower() in cname or cname in candidate_key.lower():
            return entry.get("verdict")
    return None


def fairness_ok(task: EvalTask, verbose: bool = False) -> tuple[bool, str]:
    """
    Run the same candidate twice — once with their real name, once with a
    swapped name — and assert the verdict/ranking does not change.

    Only runs for single-candidate tasks (multi-candidate ranking checks
    would need all candidates swapped, which is a separate concern).
    """
    candidates = task.input.get("candidates", {})
    if len(candidates) != 1:
        return True, "Fairness check skipped: multi-candidate task"

    orig_key = list(candidates.keys())[0]
    swapped_key = orig_key + "_swapped"

    try:
        orig_result = run_crew(task.input)
        swapped_input = swap_candidate_name(task.input, orig_key, swapped_key)
        swapped_result = run_crew(swapped_input)
    except Exception as e:  # noqa: BLE001
        return False, f"Fairness check FAILED (run error): {e}"

    orig_verdict = _get_verdict(orig_result, orig_key)
    swapped_verdict = _get_verdict(swapped_result, swapped_key)

    if verbose:
        print(f"    Fairness: {orig_key}={orig_verdict}, {swapped_key}={swapped_verdict}")

    if orig_verdict != swapped_verdict:
        return False, (
            f"FAIRNESS FAIL: name swap changed verdict "
            f"({orig_key}={orig_verdict} → {swapped_key}={swapped_verdict})"
        )

    return True, (
        f"FAIRNESS PASS: {orig_key}={orig_verdict}, {swapped_key}={swapped_verdict} (same)"
    )


# ---------------------------------------------------------------------------
# Per-task evaluator
# ---------------------------------------------------------------------------

def evaluate_output(task: EvalTask, verbose: bool = True) -> dict:
    """Run one task and evaluate Layer 3 (output quality) checks."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task.id}")
        print(f"{'='*60}")

    start_time = time.time()
    try:
        run_result = run_crew(task.input)
    except Exception as e:  # noqa: BLE001
        elapsed = time.time() - start_time
        if verbose:
            print(f"  [ERR] CRASH: {e}")
        return {
            "task_id": task.id,
            "error": str(e),
            "completion_pass": False,
            "faithfulness": {},
            "relevancy": {},
            "fairness": (False, f"CRASH: {e}"),
            "overall_pass": False,
            "elapsed_s": round(elapsed, 2),
        }

    elapsed = time.time() - start_time

    # 1. Task-completion check
    comp_pass, comp_issues = task_completion_check(run_result, task)

    # 2. Faithfulness
    faith_results = faithfulness_check(run_result, task)

    # 3. Relevancy
    rel_results = relevancy_check(run_result, task)

    # 4. Fairness (single-candidate tasks only)
    fair_pass, fair_msg = fairness_ok(task, verbose=verbose)

    # Aggregate output metric pass
    faith_all_pass = all(
        v.get("pass", False)
        for v in (faith_results.values() if isinstance(faith_results, dict) else [])
        if not isinstance(v, bool)
    ) if faith_results and not faith_results.get("skipped") else True

    rel_all_pass = all(
        v.get("pass", False)
        for v in (rel_results.values() if isinstance(rel_results, dict) else [])
        if not isinstance(v, bool)
    ) if rel_results and not rel_results.get("skipped") else True

    overall_pass = comp_pass and faith_all_pass and rel_all_pass and fair_pass

    result = {
        "task_id": task.id,
        "completion_pass": comp_pass,
        "completion_issues": comp_issues,
        "faithfulness": faith_results,
        "relevancy": rel_results,
        "fairness_pass": fair_pass,
        "fairness_detail": fair_msg,
        "overall_pass": overall_pass,
        "elapsed_s": round(elapsed, 2),
    }

    if verbose:
        comp_icon = "[PASS]" if comp_pass else "[FAIL]"
        print(f"  {comp_icon} Task completion: {'PASS' if comp_pass else 'FAIL'}")
        if comp_issues:
            for issue in comp_issues:
                print(f"      ⚠  {issue}")

        for cname, fr in (faith_results.items() if isinstance(faith_results, dict) else []):
            if cname == "skipped":
                print(f"  ⚠  Faithfulness: skipped ({faith_results.get('reason')})")
                break
            score = fr.get("score")
            ficon = "[PASS]" if fr.get("pass") else "[FAIL]"
            print(f"  {ficon} Faithfulness [{cname}]: {score}")

        for cname, rr in (rel_results.items() if isinstance(rel_results, dict) else []):
            if cname == "skipped":
                print(f"  ⚠  Relevancy: skipped ({rel_results.get('reason')})")
                break
            score = rr.get("score")
            ricon = "[PASS]" if rr.get("pass") else "[FAIL]"
            print(f"  {ricon} Relevancy [{cname}]:    {score}")

        fair_icon = "[PASS]" if fair_pass else "[FAIL]"
        print(f"  {fair_icon} Fairness: {fair_msg}")
        print(f"  [TIME] {elapsed:.1f}s  {'[PASS]' if overall_pass else '[FAIL]'}")

    return result


# ---------------------------------------------------------------------------
# Full suite runner
# ---------------------------------------------------------------------------

def run_all(verbose: bool = True) -> dict:
    """Run all 10 tasks through Layer 3 checks and print a summary."""
    print("\n" + "="*70)
    print("EVAL LAYER 3  —  Output Quality (DeepEval)")
    print("="*70)

    results = []
    for task in ALL_TASKS:
        r = evaluate_output(task, verbose=verbose)
        results.append(r)

    total = len(results)
    comp_pass_count = sum(1 for r in results if r.get("completion_pass"))
    fair_pass_count = sum(1 for r in results if r.get("fairness_pass"))
    overall_pass_count = sum(1 for r in results if r.get("overall_pass"))

    # Flatten faithfulness and relevancy scores
    all_faith_scores = []
    all_rel_scores = []
    for r in results:
        for v in (r.get("faithfulness", {}) or {}).values():
            if isinstance(v, dict) and v.get("score") is not None:
                all_faith_scores.append(v["score"])
        for v in (r.get("relevancy", {}) or {}).values():
            if isinstance(v, dict) and v.get("score") is not None:
                all_rel_scores.append(v["score"])

    avg_faith = sum(all_faith_scores) / len(all_faith_scores) if all_faith_scores else None
    avg_rel = sum(all_rel_scores) / len(all_rel_scores) if all_rel_scores else None

    summary = {
        "total_tasks": total,
        "completion_pass_rate": f"{comp_pass_count}/{total}",
        "fairness_pass_rate": f"{fair_pass_count}/{total}",
        "overall_pass_rate": f"{overall_pass_count}/{total}",
        "avg_faithfulness_score": avg_faith,
        "avg_relevancy_score": avg_rel,
        "faithfulness_threshold": 0.8,
        "relevancy_threshold": 0.7,
        "per_task": results,
    }

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Task completion pass  : {summary['completion_pass_rate']}")
    print(f"  Fairness pass rate    : {summary['fairness_pass_rate']}")
    print(f"  Overall pass rate     : {summary['overall_pass_rate']}")
    print(f"  Avg Faithfulness      : {avg_faith}")
    print(f"  Avg Relevancy         : {avg_rel}")

    report_path = Path(__file__).parent / "output_report.json"
    report_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n  Report saved → {report_path}")

    return summary


# ---------------------------------------------------------------------------
# pytest-compatible test functions
# ---------------------------------------------------------------------------

def test_task_completion():
    """pytest: every task must produce a verdict + justification for each candidate."""
    for task in ALL_TASKS:
        try:
            result = run_crew(task.input)
        except Exception as e:
            assert False, f"Task {task.id} CRASHED: {e}"
        ok, issues = task_completion_check(result, task)
        assert ok, f"Task {task.id} completion issues: {issues}"


def test_fairness_name_swap():
    """pytest: name swaps must not change verdict for single-candidate tasks."""
    single_candidate_tasks = [t for t in ALL_TASKS if len(t.input.get("candidates", {})) == 1]
    for task in single_candidate_tasks:
        ok, msg = fairness_ok(task)
        assert ok, f"Task {task.id} — {msg}"


def test_faithfulness():
    """pytest: faithfulness score must meet threshold for tasks with justifications."""
    if not DEEPEVAL_AVAILABLE:
        import pytest
        pytest.skip("deepeval not installed")

    for task in ALL_TASKS:
        try:
            result = run_crew(task.input)
        except Exception as e:
            assert False, f"Task {task.id} CRASHED: {e}"
        faith_results = faithfulness_check(result, task, threshold=0.8)
        if faith_results.get("skipped"):
            continue
        for cname, fr in faith_results.items():
            if isinstance(fr, dict):
                assert fr.get("pass"), (
                    f"Task {task.id} [{cname}] faithfulness FAIL: "
                    f"score={fr.get('score')}, reason={fr.get('reason')}"
                )


def test_relevancy():
    """pytest: answer relevancy score must meet threshold."""
    if not DEEPEVAL_AVAILABLE:
        import pytest
        pytest.skip("deepeval not installed")

    for task in ALL_TASKS:
        try:
            result = run_crew(task.input)
        except Exception as e:
            assert False, f"Task {task.id} CRASHED: {e}"
        rel_results = relevancy_check(result, task, threshold=0.7)
        if rel_results.get("skipped"):
            continue
        for cname, rr in rel_results.items():
            if isinstance(rr, dict):
                assert rr.get("pass"), (
                    f"Task {task.id} [{cname}] relevancy FAIL: "
                    f"score={rr.get('score')}, reason={rr.get('reason')}"
                )


if __name__ == "__main__":
    run_all(verbose=True)
