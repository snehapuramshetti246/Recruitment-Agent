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


# ── Main layout ───────────────────────────────────────────────────────────────

render_sidebar()

tab1, tab2 = st.tabs(["📋 Shortlist", "🔍 Trajectory & Guardrails"])

with tab1:
    render_shortlist_tab()

with tab2:
    render_trajectory_tab()
