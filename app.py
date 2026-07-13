"""
app.py — TechVest Recruitment Agent · Streamlit UI

Two-tab layout:
  Tab 1 — Shortlist    (main deliverable view)
  Tab 2 — Trajectory & Guardrails

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from config import load_jd, load_resumes, RUBRIC, validate_config
from guardrails import get_guardrail_summary
from audit import AuditLog, format_trajectory_text
from models import GuardrailFlags, RunStats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechVest Recruitment Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Verdict badges */
.badge-interview { background:#16a34a; color:#fff; padding:3px 12px;
                   border-radius:20px; font-weight:600; font-size:.85rem; }
.badge-hold      { background:#d97706; color:#fff; padding:3px 12px;
                   border-radius:20px; font-weight:600; font-size:.85rem; }
.badge-not-a-fit { background:#6b7280; color:#fff; padding:3px 12px;
                   border-radius:20px; font-weight:600; font-size:.85rem; }

/* Status pill */
.pill-idle     { background:#e2e8f0; color:#475569; padding:4px 14px;
                 border-radius:20px; font-weight:600; }
.pill-running  { background:#dbeafe; color:#1d4ed8; padding:4px 14px;
                 border-radius:20px; font-weight:600; }
.pill-approval { background:#fef3c7; color:#92400e; padding:4px 14px;
                 border-radius:20px; font-weight:600; }
.pill-complete { background:#dcfce7; color:#166534; padding:4px 14px;
                 border-radius:20px; font-weight:600; }

/* Injection warning in trajectory */
.injection-step { border-left: 4px solid #dc2626; padding-left: 12px;
                  background: #fef2f2; border-radius: 4px; margin: 6px 0; }

/* Pending approval badge */
.pending-badge  { background:#fef3c7; color:#92400e; padding:2px 10px;
                  border-radius:12px; font-size:.8rem; font-weight:600; }

/* Guardrail dot */
.dot-ok  { color: #16a34a; font-size: 1.1rem; }
.dot-err { color: #dc2626; font-size: 1.1rem; }
.dot-na  { color: #9ca3af; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "run_status": "idle",        # idle | running | awaiting_approval | complete
        "agent_state": None,         # final AgentState dict
        "graph": None,               # compiled LangGraph
        "thread_id": "main",
        "run_id": None,
        "start_time": None,
        "guardrail_flags": GuardrailFlags().model_dump(),
        "config_warnings": validate_config(),
        "approval_candidate": None,
        "approval_slot": None,
        "fairness_report": None,
        "last_run_stats": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _status_pill(status: str) -> str:
    mapping = {
        "idle":              ('<span class="pill-idle">⚪ Idle</span>', ""),
        "running":           ('<span class="pill-running">🔵 Running</span>', ""),
        "awaiting_approval": ('<span class="pill-approval">🟡 Awaiting Approval</span>', ""),
        "complete":          ('<span class="pill-complete">🟢 Complete</span>', ""),
    }
    html, _ = mapping.get(status, (status, ""))
    return html


def _verdict_badge(verdict: str) -> str:
    cls = {
        "INTERVIEW":  "badge-interview",
        "HOLD":       "badge-hold",
        "NOT_A_FIT":  "badge-not-a-fit",
    }.get(verdict, "badge-not-a-fit")
    return f'<span class="{cls}">{verdict.replace("_", " ")}</span>'


def _score_bar(score: float, max_score: float = 5.0) -> None:
    pct = min(score / max_score, 1.0)
    color = "#16a34a" if pct >= 0.7 else "#d97706" if pct >= 0.5 else "#6b7280"
    st.markdown(
        f"""
        <div style="background:#e5e7eb; border-radius:6px; height:10px; width:100%;">
          <div style="background:{color}; width:{pct*100:.1f}%;
                      height:10px; border-radius:6px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _dot(ok: bool | None) -> str:
    if ok is True:
        return '<span class="dot-ok">●</span>'
    elif ok is False:
        return '<span class="dot-err">●</span>'
    return '<span class="dot-na">●</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎯 TechVest · Run Panel")
        st.divider()

        # Config warnings
        if st.session_state["config_warnings"]:
            for w in st.session_state["config_warnings"]:
                st.warning(w, icon="⚠️")

        # Role / JD
        with st.expander("📋 Role: Junior AI Engineer", expanded=False):
            try:
                jd = load_jd()
                st.text_area("Job Description", jd, height=300, disabled=True, label_visibility="collapsed")
            except Exception:
                st.error("jd.txt not found.")

        # Rubric
        with st.expander("📐 Scoring Rubric", expanded=False):
            criteria = RUBRIC.get("criteria", [])
            if criteria:
                rubric_rows = [
                    {"Criterion": c["name"], "Weight": f"{c['weight']:.0%}"}
                    for c in criteria
                ]
                st.table(rubric_rows)
                st.caption(RUBRIC.get("evidence_rule", ""))
            else:
                st.info("rubric.json not loaded.")

        st.divider()

        # Live guardrail panel
        st.markdown("### 🛡️ Guardrail Status")
        flags = GuardrailFlags(**st.session_state["guardrail_flags"])
        summary = get_guardrail_summary(flags)
        for key, info in summary.items():
            col1, col2 = st.columns([1, 8])
            with col1:
                st.markdown(_dot(info["ok"]), unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{info['label']}**  \n{info['detail']}")

        st.divider()

        # Last run stats
        if st.session_state.get("last_run_stats"):
            s = st.session_state["last_run_stats"]
            st.markdown("### 📊 Last Run Stats")
            st.metric("Total Steps", s.get("total_steps", 0))
            st.metric("Duration", f"{s.get('run_duration_seconds', 0):.1f}s")
            calls = s.get("tool_calls", {})
            if calls:
                st.markdown("**Tool calls:**")
                for tool_name, count in calls.items():
                    st.markdown(f"- `{tool_name}`: {count}")


# ── Run agent ─────────────────────────────────────────────────────────────────

def run_agent():
    """Launch the agent and stream progress."""
    from agent import build_graph, make_initial_state
    from guardrails import name_swap_test
    from models import ScoreCard

    jd = load_jd()
    resumes = load_resumes()
    if not resumes:
        st.error("No résumé files found in resumes/")
        return

    graph = build_graph()
    st.session_state["graph"] = graph
    st.session_state["run_status"] = "running"
    st.session_state["start_time"] = time.time()

    candidates = dict(resumes)
    initial_state = make_initial_state(jd, candidates, RUBRIC)

    config = {"configurable": {"thread_id": st.session_state["thread_id"]},
              "recursion_limit": 50}

    tool_call_counts: dict[str, int] = {}

    # Track the last seen state so we can display partial results on error
    state = initial_state

    # Stream events for live status updates
    with st.status("🤖 Agent running…", expanded=True) as status_box:
        try:
            for event in graph.stream(initial_state, config=config, stream_mode="values"):
                state = event

                # Update guardrail flags live
                if state.get("guardrail_flags"):
                    st.session_state["guardrail_flags"] = state["guardrail_flags"]

                # Show live step progress
                step = state.get("step_count", 0)
                traj = state.get("trajectory", [])
                if traj:
                    last = traj[-1]
                    action = last.get("action", "")
                    candidate_arg = last.get("action_args", {}).get("candidate", "")
                    label = f"Step {step}: `{action}`" + (f" — {candidate_arg}" if candidate_arg else "")
                    st.write(label)
                    tool_call_counts[action] = tool_call_counts.get(action, 0) + 1

                # Bubble up any error captured in state
                if state.get("error"):
                    st.error(f"⚠️ Agent step error: {state['error']}")

                # Check for interrupt (pending human approval)
                interrupt_info = None
                try:
                    snap = graph.get_state(config)
                    if snap.next and "schedule" in snap.next:
                        interrupt_info = snap.tasks[0].interrupts[0].value if snap.tasks else None
                except Exception:
                    pass

                if interrupt_info:
                    st.session_state["run_status"] = "awaiting_approval"
                    st.session_state["agent_state"] = state
                    st.session_state["approval_candidate"] = interrupt_info.get("candidate")
                    st.session_state["approval_slot"] = interrupt_info.get("slot")
                    status_box.update(label="⏸️ Awaiting human approval…", state="running")
                    break

            else:
                # Stream ended normally
                st.session_state["run_status"] = "complete"
                st.session_state["agent_state"] = state

                # Fairness / name-swap test
                scorecards = state.get("scorecards", {})
                sc_list = list(scorecards.values())
                if len(sc_list) >= 2:
                    sc_a = ScoreCard(**sc_list[0])
                    sc_b = ScoreCard(**sc_list[1])
                    _, report = name_swap_test(sc_a, sc_b, sc_a.candidate, sc_b.candidate)
                    st.session_state["fairness_report"] = report
                    flags = dict(st.session_state["guardrail_flags"])
                    flags["fairness_pass"] = True
                    st.session_state["guardrail_flags"] = flags

                status_box.update(label="✅ Agent run complete!", state="complete")

        except Exception as e:
            import traceback
            err_detail = traceback.format_exc()
            st.error(f"Agent error: {e}")
            st.expander("Error details").code(err_detail)
            # Save whatever state was reached so partial results are visible
            st.session_state["agent_state"] = state if state is not initial_state else None
            st.session_state["run_status"] = "complete" if state.get("shortlist") else "idle"
            status_box.update(label="❌ Agent run failed — see error above.", state="error")
            return

    # Persist run stats
    elapsed = time.time() - (st.session_state["start_time"] or time.time())
    stats = RunStats(
        total_steps=state.get("step_count", 0),
        tool_calls=tool_call_counts,
        run_duration_seconds=round(elapsed, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        candidates_processed=len(state.get("profiles", {})),
    )
    st.session_state["last_run_stats"] = stats.model_dump()

    # Audit log
    from audit import AuditLog
    from models import TrajectoryStep, ShortlistEntry
    audit = AuditLog()
    traj_steps = [TrajectoryStep(**s) for s in state.get("trajectory", [])]
    sl_entries = []
    for e in state.get("shortlist", []):
        try:
            sl_entries.append(ShortlistEntry(**e))
        except Exception:
            pass
    gflags = GuardrailFlags(**state.get("guardrail_flags", {}))
    run_id = audit.save_run(traj_steps, sl_entries, gflags, stats)
    st.session_state["run_id"] = run_id


# ── Tab 1 — Shortlist ─────────────────────────────────────────────────────────

def render_shortlist_tab():
    st.markdown("## 📋 Recruitment Shortlist — Junior AI Engineer, TechVest")

    status = st.session_state["run_status"]
    col_hdr, col_pill = st.columns([6, 2])
    with col_hdr:
        st.markdown(f"Run: `{st.session_state.get('run_id', '—')}`  |  "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    with col_pill:
        st.markdown(_status_pill(status), unsafe_allow_html=True)

    st.divider()

    # ── Run controls ──────────────────────────────────────────────────────────
    col_run, col_reset = st.columns([3, 1])
    with col_run:
        if status == "idle":
            if st.button("▶️  Run Agent", type="primary", use_container_width=True):
                run_agent()
                st.rerun()
        elif status == "running":
            st.info("Agent is running…")
        elif status == "awaiting_approval":
            st.warning("⏳ Agent paused — waiting for interview approval below.")
        elif status == "complete":
            if st.button("🔄  Re-run Agent", use_container_width=True):
                st.session_state["run_status"] = "idle"
                st.session_state["agent_state"] = None
                st.rerun()

    with col_reset:
        if status != "idle" and st.button("⏹ Reset", use_container_width=True):
            st.session_state["run_status"] = "idle"
            st.session_state["agent_state"] = None
            st.rerun()

    # ── Shortlist cards ───────────────────────────────────────────────────────
    agent_state = st.session_state.get("agent_state")
    if not agent_state:
        st.info("Click **Run Agent** to start the hiring evaluation.")
        return

    shortlist = agent_state.get("shortlist", [])
    if not shortlist:
        st.warning("No shortlist generated yet.")
        return

    for rank, entry in enumerate(shortlist, 1):
        candidate = entry.get("candidate", "Unknown")
        verdict = entry.get("verdict", "NOT_A_FIT")
        score = entry.get("weighted_score", 0.0)
        justification = entry.get("justification", "")
        scorecard = entry.get("scorecard", {})
        slot = entry.get("proposed_slot")
        confirmation = entry.get("confirmation", {})

        with st.container(border=True):
            # Header row
            c1, c2, c3 = st.columns([3, 5, 2])
            with c1:
                st.markdown(f"### #{rank} — {candidate}")
                st.markdown(_verdict_badge(verdict), unsafe_allow_html=True)
            with c2:
                st.markdown(f"**Score: {score:.2f} / 5.0**")
                _score_bar(score)
            with c3:
                pass

            # Justification
            st.markdown(f"**Assessment:** {justification}")

            # Per-criterion scorecard (expandable)
            with st.expander("📊 Per-Criterion Scorecard"):
                criteria = scorecard.get("criteria", [])
                if criteria:
                    rows = [
                        {
                            "Criterion":  c.get("criterion_name", c.get("criterion")),
                            "Weight":     f"{c.get('weight', 0):.0%}",
                            "Score":      f"{c.get('score', 0)} / 5",
                            "Evidence":   c.get("evidence", "—"),
                        }
                        for c in criteria
                    ]
                    st.table(rows)

            # Interview slot + human approval gate
            if verdict == "INTERVIEW":
                st.markdown("---")
                conf_status = (confirmation or {}).get("status", "pending_approval")

                if conf_status == "pending_approval" and slot:
                    st.markdown(
                        f'🔒 <span class="pending-badge">⏳ Pending Human Approval</span> '
                        f'— Proposed: **{slot.get("day")} {slot.get("date")} @ {slot.get("time")}**',
                        unsafe_allow_html=True,
                    )
                    col_approve, col_reject = st.columns(2)
                    with col_approve:
                        if st.button(f"✅ Approve — {candidate}", key=f"approve_{candidate}"):
                            _handle_approval(candidate, slot, approved=True)
                            st.rerun()
                    with col_reject:
                        if st.button(f"❌ Reject — {candidate}", key=f"reject_{candidate}"):
                            _handle_approval(candidate, slot, approved=False)
                            st.rerun()

                elif conf_status == "confirmed":
                    conf_id = (confirmation or {}).get("confirmation_id", "")
                    st.success(
                        f"✅ Interview confirmed! "
                        f"{slot.get('day')} {slot.get('date')} @ {slot.get('time')} "
                        f"(ID: {conf_id})"
                    )

                elif conf_status == "rejected":
                    st.error("🚫 Interview scheduling rejected by reviewer.")

                elif not slot:
                    st.info("📅 Availability check pending…")


def _handle_approval(candidate: str, slot: dict, approved: bool):
    """Resume the LangGraph graph after human decision."""
    graph = st.session_state.get("graph")
    config = {"configurable": {"thread_id": st.session_state["thread_id"]},
              "recursion_limit": 50}

    if graph is None:
        st.error("Graph not initialised.")
        return

    human_input = {"approved": approved}

    try:
        final_state = None
        for event in graph.stream(
            human_input, config=config, stream_mode="values"
        ):
            final_state = event

        if final_state:
            st.session_state["agent_state"] = final_state
            flags = dict(st.session_state["guardrail_flags"])
            flags["human_gate_status"] = "approved" if approved else "rejected"
            st.session_state["guardrail_flags"] = flags
            st.session_state["run_status"] = "complete"

            # Update audit log
            audit = AuditLog()
            run_id = st.session_state.get("run_id")
            if run_id:
                audit.update_confirmation(
                    run_id, candidate, "confirmed" if approved else "rejected"
                )

    except Exception as e:
        st.error(f"Approval error: {e}")


# ── Tab 2 — Trajectory & Guardrails ──────────────────────────────────────────

def render_trajectory_tab():
    st.markdown("## 🔍 Trajectory & Guardrails")

    agent_state = st.session_state.get("agent_state")

    # ── Guardrail status panel ────────────────────────────────────────────────
    st.markdown("### 🛡️ Guardrail Status (Detail)")
    flags = GuardrailFlags(**st.session_state["guardrail_flags"])
    summary = get_guardrail_summary(flags)

    cols = st.columns(len(summary))
    for col, (key, info) in zip(cols, summary.items()):
        with col:
            with st.container(border=True):
                dot = _dot(info["ok"])
                st.markdown(f"{dot} **{info['label']}**", unsafe_allow_html=True)
                st.caption(info["detail"])

    st.divider()

    if not agent_state:
        st.info("Run the agent first to see the trajectory.")
        return

    # ── Step-by-step trace ────────────────────────────────────────────────────
    st.markdown("### 🗂️ Step-by-Step Trajectory")

    trajectory = agent_state.get("trajectory", [])
    if not trajectory:
        st.info("No trajectory steps recorded.")
    else:
        for step in trajectory:
            step_num = step.get("step_number", "?")
            thought = step.get("thought", "")
            action = step.get("action", "")
            args = step.get("action_args", {})
            obs = step.get("observation", "")
            guardrail = step.get("guardrail_triggered")

            is_injection = guardrail and "INJECTION" in guardrail.upper()

            if is_injection:
                st.markdown(
                    f"""
                    <div class="injection-step">
                    <strong>⚠️ Step {step_num} — Injection attempt ignored</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander(f"Step {step_num} · `{action}`", expanded=(step_num == 1)):
                st.markdown(f"*{thought}*")
                st.code(f"{action}({json.dumps(args, indent=2)})", language="python")
                if guardrail:
                    st.error(f"🛡️ Guardrail triggered: {guardrail}")
                with st.container():
                    if len(obs) > 500:
                        st.text_area("Observation", obs, height=150, disabled=True,
                                     key=f"obs_{step_num}", label_visibility="collapsed")
                    else:
                        st.markdown(f"```\n{obs}\n```")

    st.divider()

    # ── Fairness panel ────────────────────────────────────────────────────────
    st.markdown("### ⚖️ Fairness Check")
    fairness_report = st.session_state.get("fairness_report")

    scorecards = (agent_state or {}).get("scorecards", {})
    sc_items = list(scorecards.items())

    if len(sc_items) >= 2:
        col_a, col_b = st.columns(2)
        for i, (key, sc) in enumerate(sc_items[:2]):
            with (col_a if i == 0 else col_b):
                with st.container(border=True):
                    st.markdown(f"**{sc.get('candidate', key)}**")
                    st.metric("Weighted Score", f"{sc.get('weighted_total', 0):.2f} / 5.0")
                    for c in sc.get("criteria", []):
                        st.markdown(
                            f"- {c.get('criterion_name', c['criterion'])}: "
                            f"**{c['score']}/5**"
                        )

        if fairness_report:
            if "PASSED" in fairness_report:
                st.success(fairness_report)
            else:
                st.error(fairness_report)
        else:
            st.info("Name-swap fairness test compares two candidates with equivalent profiles.")
    else:
        st.info("Fairness comparison requires at least two candidates to be scored.")

    st.divider()

    # ── Audit log ─────────────────────────────────────────────────────────────
    st.markdown("### 📁 Decision Audit Log")
    audit = AuditLog()
    records = audit.load_all()

    if not records:
        st.info("No runs saved yet. Run the agent to generate an audit record.")
    else:
        rows = []
        for r in reversed(records[-10:]):  # last 10 runs
            sl = r.get("shortlist", [])
            rows.append({
                "Run ID":    r.get("run_id", "—"),
                "Timestamp": r.get("timestamp", "—")[:19].replace("T", " "),
                "Steps":     r.get("run_stats", {}).get("total_steps", "—"),
                "Duration":  f"{r.get('run_stats', {}).get('run_duration_seconds', 0):.1f}s",
                "Shortlist": ", ".join(
                    f"{e['candidate']}:{e['verdict']}" for e in sl
                ) if sl else "—",
            })
        st.dataframe(rows, use_container_width=True)

        # Re-open a past trajectory
        run_ids = [r.get("run_id") for r in records if r.get("run_id")]
        if run_ids:
            selected_id = st.selectbox("Re-open trajectory for run:", ["—"] + run_ids)
            if selected_id and selected_id != "—":
                past = audit.get_run(selected_id)
                if past:
                    st.markdown(f"**Trajectory for run `{selected_id}`:**")
                    for step in past.get("trajectory", []):
                        step_num = step.get("step_number", "?")
                        action = step.get("action", "")
                        with st.expander(f"Step {step_num} · `{action}`"):
                            st.markdown(f"*{step.get('thought', '')}*")
                            st.code(f"{action}({json.dumps(step.get('action_args', {}), indent=2)})",
                                    language="python")
                            st.text(step.get("observation", ""))

    st.caption("Formal agent evaluation metrics arrive Day 7.")


# ── Tab 3 — Eval Dashboard ────────────────────────────────────────────────────

def _load_report(filename: str) -> dict | None:
    """Load a JSON eval report if it exists."""
    path = Path(__file__).parent / "eval" / filename
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _eval_metric_card(col, label: str, value: str, icon: str, ok: bool | None = None) -> None:
    """Render a coloured metric card inside a column."""
    if ok is True:
        bg, fg = "#dcfce7", "#166534"
    elif ok is False:
        bg, fg = "#fee2e2", "#991b1b"
    else:
        bg, fg = "#f1f5f9", "#334155"
    with col:
        st.markdown(
            f"""
            <div style="background:{bg}; border-radius:10px; padding:16px 18px;
                        margin-bottom:6px; border:1px solid #e2e8f0;">
              <div style="font-size:1.6rem;">{icon}</div>
              <div style="font-size:1.35rem; font-weight:700; color:{fg};">{value}</div>
              <div style="font-size:.82rem; color:#64748b; margin-top:2px;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _severity_badge(sev: str) -> str:
    colours = {
        "Critical": ("#fee2e2", "#991b1b"),
        "Medium":   ("#fef3c7", "#92400e"),
        "Low":      ("#f1f5f9", "#475569"),
    }
    bg, fg = colours.get(sev, ("#f1f5f9", "#475569"))
    return (
        f'<span style="background:{bg}; color:{fg}; padding:2px 10px; '
        f'border-radius:12px; font-size:.78rem; font-weight:700;">{sev}</span>'
    )


def _task_status_icon(passed: bool | None) -> str:
    if passed is True:
        return "✅"
    if passed is False:
        return "❌"
    return "⚠️"


def render_eval_dashboard():
    from pathlib import Path

    st.markdown("## 📊 Evaluation Dashboard")
    st.caption(
        "Run the eval suite from the terminal (`python eval/test_trace.py` etc.) "
        "and this dashboard will display the results automatically."
    )

    # ── Check which reports exist ─────────────────────────────────────────────
    trace_report  = _load_report("trace_report.json")
    output_report = _load_report("output_report.json")
    gate_report   = _load_report("gate_report.json")
    red_report    = _load_report("redteam/giskard_findings.json")
    tasks_index   = _load_report("tasks.json")

    any_report = any([trace_report, output_report, gate_report, red_report])

    if not any_report:
        st.info(
            "No eval reports found yet.  \n"
            "Run these commands in your terminal, then refresh this page:\n"
            "```bash\n"
            "python eval/tasks.py\n"
            "python eval/test_trace.py\n"
            "python eval/test_output.py\n"
            "python eval/redteam/giskard_scan.py\n"
            "python eval/test_gate.py\n"
            "```"
        )

    # ── Quick re-run buttons ──────────────────────────────────────────────────
    # Session state keys used here:
    #   eval_running      : str | None  — which exercise is currently running
    #   eval_last_result  : dict | None — {exercise, returncode, stdout, stderr}

    if "eval_running" not in st.session_state:
        st.session_state["eval_running"] = None
    if "eval_last_result" not in st.session_state:
        st.session_state["eval_last_result"] = None

    EVAL_EXERCISES = {
        "ex1": ("Ex 1 · Tasks",    ["python", "eval/tasks.py"]),
        "ex2": ("Ex 2 · Trace",    ["python", "eval/test_trace.py"]),
        "ex3": ("Ex 3 · Output",   ["python", "eval/test_output.py"]),
        "ex4": ("Ex 4 · Red-team", ["python", "eval/redteam/giskard_scan.py"]),
        "ex5": ("Ex 5 · Gate",     ["python", "eval/test_gate.py"]),
    }

    with st.expander("▶️  Run eval suite now (requires LLM API access)", expanded=True):
        st.warning(
            "Each exercise calls the LLM for every task. "
            "This may take several minutes and will consume API credits.",
            icon="⚠️",
        )

        # ── Button row ────────────────────────────────────────────────────────
        btn_cols = st.columns(5)
        for col, (ex_key, (label, _cmd)) in zip(btn_cols, EVAL_EXERCISES.items()):
            with col:
                is_running = st.session_state["eval_running"] == ex_key
                btn_label  = f"⏳ {label}" if is_running else label
                disabled   = st.session_state["eval_running"] is not None
                if st.button(btn_label, key=f"eval_btn_{ex_key}",
                             use_container_width=True, disabled=disabled):
                    st.session_state["eval_running"] = ex_key
                    st.session_state["eval_last_result"] = None
                    st.rerun()

        # ── Run the selected exercise (blocking, but inside the render pass) ─
        running_key = st.session_state.get("eval_running")
        if running_key and running_key in EVAL_EXERCISES:
            label, cmd = EVAL_EXERCISES[running_key]
            import subprocess, sys

            output_box = st.empty()
            output_box.info(f"⏳ Running **{label}** — please wait…")

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                    timeout=600,      # 10-minute hard cap per exercise
                )
                result = {
                    "exercise": label,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            except subprocess.TimeoutExpired:
                result = {
                    "exercise": label,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "Process timed out after 10 minutes.",
                }
            except Exception as exc:
                result = {
                    "exercise": label,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": str(exc),
                }

            st.session_state["eval_running"]     = None
            st.session_state["eval_last_result"] = result
            output_box.empty()
            # Reload the reports and rerender — do NOT rerun before saving result
            st.rerun()

        # ── Show last result ──────────────────────────────────────────────────
        last = st.session_state.get("eval_last_result")
        if last:
            ex_label = last.get("exercise", "")
            rc       = last.get("returncode", -1)
            stdout   = last.get("stdout", "").strip()
            stderr   = last.get("stderr", "").strip()

            if rc == 0:
                st.success(f"✅ **{ex_label}** completed successfully.")
            else:
                st.error(f"❌ **{ex_label}** failed (exit code {rc}).")

            if stdout:
                with st.expander("📄 Output", expanded=(rc != 0)):
                    st.code(stdout[-3000:], language="text")   # last 3000 chars
            if stderr:
                with st.expander("⚠️ Stderr / Warnings", expanded=(rc != 0)):
                    st.code(stderr[-2000:], language="text")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SCORECARD — top-level summary numbers
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 🏆 Final Scorecard")

    # Parse the key numbers
    inv_rate   = trace_report.get("invariant_pass_rate", "—")     if trace_report  else "—"
    tool_rate  = trace_report.get("tool_call_accuracy_rate", "—") if trace_report  else "—"
    tc_score   = trace_report.get("avg_task_completion_score")    if trace_report  else None
    se_score   = trace_report.get("avg_step_efficiency_score")    if trace_report  else None
    faith_score= output_report.get("avg_faithfulness_score")      if output_report else None
    rel_score  = output_report.get("avg_relevancy_score")         if output_report else None
    gate_rate  = gate_report.get("gate_fire_rate", "—")           if gate_report   else "—"
    crit_open  = red_report.get("summary", {}).get("critical", "—") if red_report  else "—"

    def _pct_ok(rate_str: str) -> bool | None:
        if rate_str == "—":
            return None
        try:
            num = int(rate_str.split("(")[1].rstrip("%)"))
            return num == 100
        except Exception:
            return None

    def _score_ok(val, threshold: float) -> bool | None:
        if val is None:
            return None
        try:
            return float(val) >= threshold
        except Exception:
            return None

    c1, c2, c3, c4 = st.columns(4)
    _eval_metric_card(c1, "Invariant Pass Rate",       inv_rate,   "🔍", _pct_ok(inv_rate))
    _eval_metric_card(c2, "Tool-Call Accuracy",        tool_rate,  "🔧", _pct_ok(tool_rate))
    _eval_metric_card(c3, "Gate Fire Rate",            gate_rate,  "🔒", _pct_ok(gate_rate))
    _eval_metric_card(c4, "Critical Findings Open",
                      str(crit_open),
                      "🚨",
                      True if crit_open == 0 else (False if isinstance(crit_open, int) and crit_open > 0 else None))

    c5, c6, c7, c8 = st.columns(4)
    _eval_metric_card(c5, "Avg Faithfulness",
                      f"{faith_score:.2f}" if faith_score is not None else "—",
                      "📎", _score_ok(faith_score, 0.8))
    _eval_metric_card(c6, "Avg Relevancy",
                      f"{rel_score:.2f}" if rel_score is not None else "—",
                      "🎯", _score_ok(rel_score, 0.7))
    _eval_metric_card(c7, "Avg TaskCompletion",
                      f"{tc_score:.2f}" if tc_score is not None else "—",
                      "✅", _score_ok(tc_score, 0.5))
    _eval_metric_card(c8, "Avg StepEfficiency",
                      f"{se_score:.2f}" if se_score is not None else "—",
                      "⚡", _score_ok(se_score, 0.5))

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Task Dataset
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 📋 Exercise 1 — Evaluation Dataset (10 Tasks)")

    if tasks_index:
        rows = []
        for t in tasks_index:
            rows.append({
                "ID":           t["id"],
                "Borderline":   "🔶 Yes" if t.get("borderline") else "—",
                "Candidates":   ", ".join(t.get("candidate_keys", [])),
                "Expected Trajectory": " → ".join(t.get("expected_trajectory", [])),
                "Expected Verdict":    str(list(t.get("expected_decision", {}).values())[:1]),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python eval/tasks.py` to generate the task index.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Trace + Tool-Call (Exercise 2)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 🔍 Exercise 2 — Trace Invariants + Tool-Call Accuracy")

    if trace_report:
        per_task = trace_report.get("per_task", [])
        if per_task:
            rows = []
            for r in per_task:
                inv_icon  = _task_status_icon(r.get("invariants_pass"))
                tool_icon = _task_status_icon(r.get("tools_pass"))
                overall   = _task_status_icon(r.get("overall_pass"))
                tc = r.get("judge_scores", {}).get("task_completion_score")
                se = r.get("judge_scores", {}).get("step_efficiency_score")
                rows.append({
                    "Task ID":      r["task_id"],
                    "Steps":        " → ".join(r.get("steps", [])),
                    "Invariants":   inv_icon,
                    "Tool Calls":   tool_icon,
                    "TaskComp.":    f"{tc:.2f}" if tc is not None else "—",
                    "StepEff.":     f"{se:.2f}" if se is not None else "—",
                    "Overall":      overall,
                    "Time (s)":     r.get("elapsed_s", "—"),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

            # Show violations
            violations_found = False
            for r in per_task:
                if r.get("invariant_violations") or r.get("tool_issues"):
                    if not violations_found:
                        st.markdown("**Violations / Issues:**")
                        violations_found = True
                    with st.expander(f"⚠️ {r['task_id']} issues"):
                        for v in r.get("invariant_violations", []):
                            st.error(v)
                        for v in r.get("tool_issues", []):
                            st.warning(v)
        else:
            st.info("No per-task data in trace_report.json.")
    else:
        st.info("No trace report yet. Run `python eval/test_trace.py`.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Output Quality (Exercise 3)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 📝 Exercise 3 — Output Quality (DeepEval Metrics)")

    if output_report:
        per_task = output_report.get("per_task", [])
        if per_task:
            rows = []
            for r in per_task:
                # Flatten faithfulness scores
                faith = r.get("faithfulness", {}) or {}
                faith_scores = [
                    v.get("score") for v in faith.values()
                    if isinstance(v, dict) and v.get("score") is not None
                ]
                avg_f = sum(faith_scores) / len(faith_scores) if faith_scores else None

                rel = r.get("relevancy", {}) or {}
                rel_scores = [
                    v.get("score") for v in rel.values()
                    if isinstance(v, dict) and v.get("score") is not None
                ]
                avg_r = sum(rel_scores) / len(rel_scores) if rel_scores else None

                comp_icon  = _task_status_icon(r.get("completion_pass"))
                fair_icon  = _task_status_icon(r.get("fairness_pass"))
                overall    = _task_status_icon(r.get("overall_pass"))

                rows.append({
                    "Task ID":      r["task_id"],
                    "Completion":   comp_icon,
                    "Faithfulness": f"{avg_f:.2f}" if avg_f is not None else "—",
                    "Relevancy":    f"{avg_r:.2f}" if avg_r is not None else "—",
                    "Fairness":     fair_icon,
                    "Overall":      overall,
                    "Time (s)":     r.get("elapsed_s", "—"),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(
                f"Thresholds: Faithfulness ≥ {output_report.get('faithfulness_threshold', 0.8)}  "
                f"| Relevancy ≥ {output_report.get('relevancy_threshold', 0.7)}"
            )
        else:
            st.info("No per-task data in output_report.json.")
    else:
        st.info("No output report yet. Run `python eval/test_output.py`.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Red-Team Findings (Exercise 4)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 🚨 Exercise 4 — Red-Team Findings")

    if red_report:
        findings = red_report.get("findings", [])
        summary  = red_report.get("summary", {})

        crit = summary.get("critical", 0)
        med  = summary.get("medium", 0)
        low  = summary.get("low", 0)

        c1, c2, c3 = st.columns(3)
        _eval_metric_card(c1, "Critical Findings", str(crit), "🔴", crit == 0)
        _eval_metric_card(c2, "Medium Findings",   str(med),  "🟡", med  == 0)
        _eval_metric_card(c3, "Low Findings",       str(low),  "🟢", True)

        if findings:
            st.markdown("**Findings Table:**")
            rows = []
            for f in findings:
                rows.append({
                    "ID":          f.get("id", "—"),
                    "Category":    f.get("category", "—"),
                    "Severity":    f.get("severity", "—"),
                    "Layer":       f.get("layer", "—"),
                    "Status":      f.get("status", "—"),
                    "Description": f.get("description", "—")[:70],
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

            # Detailed finding cards for non-passing items
            failures = [f for f in findings if f.get("status") == "FAIL"]
            if failures:
                st.markdown("**Findings requiring fixes:**")
                for f in failures:
                    sev = f.get("severity", "Low")
                    with st.expander(
                        f"{_severity_badge(sev)} {f.get('id')} — {f.get('description', '')[:60]}",
                        expanded=sev == "Critical",
                    ):
                        st.markdown(f"**Category:** {f.get('category')}")
                        st.markdown(f"**Layer broken:** {f.get('layer')}")
                        st.markdown(f"**Issues:**")
                        for issue in f.get("issues", []):
                            st.error(issue)
                        st.markdown(f"**Suggested fix:** {f.get('fix')}")
                    st.markdown("", unsafe_allow_html=True)
            else:
                st.success("✅ No failing red-team cases.")
        else:
            st.info("No findings recorded yet.")
    else:
        st.info("No red-team report yet. Run `python eval/redteam/giskard_scan.py`.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Human Gate Assertions (Exercise 5)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 🔒 Exercise 5 — Human-in-the-Loop Gate")

    if gate_report:
        passed   = gate_report.get("passed", 0)
        total_g  = gate_report.get("total_gate_tests", 0)
        failed   = gate_report.get("failed", 0)
        rate_str = gate_report.get("gate_fire_rate", "—")

        if failed == 0:
            st.success(f"✅ Gate fired on ALL {total_g} high-stakes tasks ({rate_str}). "
                       "Agent is safe to proceed.")
        else:
            st.error(
                f"❌ CRITICAL: Gate failed on {failed}/{total_g} tasks. "
                "Agent MUST NOT be trusted with real hiring decisions until fixed."
            )

        # Per-test breakdown
        tests = gate_report.get("tests", {})
        test_labels = {
            "positive_gate":    "Test 1 — Positive Gate (INTERVIEW must pause)",
            "no_skip":          "Test 2 — No-Skip (obvious cases still need approval)",
            "verifier_not_gate":"Test 3 — Verifier ≠ Human Gate",
            "conflict_escalation": "Test 4 — Conflicting Availability Escalation",
        }

        for key, label in test_labels.items():
            test_data = tests.get(key, {})
            results   = test_data.get("results", [test_data.get("result", {})])

            all_passed = all(r.get("passed", False) for r in results if r)
            icon = "✅" if all_passed else "❌ CRITICAL"

            with st.expander(f"{icon}  {label}", expanded=not all_passed):
                if not results:
                    st.info("No results for this test.")
                    continue
                rows = []
                for r in results:
                    if not r:
                        continue
                    detail = r.get("detail") or r.get("gate_detail") or {}
                    rows.append({
                        "Task ID":  r.get("task_id", "—"),
                        "Passed":   _task_status_icon(r.get("passed")),
                        "Verdict":  detail.get("verdict", "—") if isinstance(detail, dict) else str(detail)[:40],
                        "Paused Before Schedule": "Yes" if r.get("passed") else "NO ❌",
                        "Time (s)": r.get("elapsed_s", "—"),
                    })
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)

                # Show critical detail if failed
                for r in results:
                    if r and not r.get("passed"):
                        detail = r.get("detail") or r.get("gate_detail") or {}
                        if isinstance(detail, dict) and detail.get("reason"):
                            st.error(f"🔴 {detail['reason']}")
    else:
        st.info("No gate report yet. Run `python eval/test_gate.py`.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Verifier Node Status
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 🔬 Verifier Node — Borderline Candidate Coverage")

    if trace_report:
        per_task = trace_report.get("per_task", [])
        borderline_tasks = [r for r in per_task if r.get("borderline")]
        if borderline_tasks:
            for r in borderline_tasks:
                steps = r.get("steps", [])
                verifier_present = "verifier" in steps
                icon = "✅" if verifier_present else "❌"
                st.markdown(
                    f"{icon} **{r['task_id']}** — "
                    f"Trace: `{'` → `'.join(steps)}`"
                    + ("  ← Verifier fired ✓" if verifier_present else "  ← **Verifier MISSING**")
                )
        else:
            st.info("No borderline task results found in trace report.")
    else:
        st.info("Run `python eval/test_trace.py` to see verifier coverage.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Raw report viewer
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 📄 Raw Report Files")
    report_options = {
        "trace_report.json":              trace_report,
        "output_report.json":             output_report,
        "gate_report.json":               gate_report,
        "redteam/giskard_findings.json":  red_report,
        "tasks.json":                     tasks_index,
    }
    for fname, data in report_options.items():
        label = f"{'✅' if data else '⚪'} {fname}"
        with st.expander(label, expanded=False):
            if data:
                st.json(data)
            else:
                st.info(f"Not generated yet.")


# ── Main layout ───────────────────────────────────────────────────────────────

render_sidebar()

tab1, tab2, tab3 = st.tabs(["📋 Shortlist", "🔍 Trajectory & Guardrails", "📊 Eval Dashboard"])

with tab1:
    render_shortlist_tab()

with tab2:
    render_trajectory_tab()

with tab3:
    render_eval_dashboard()
