# Start with: python run_local.py
from __future__ import annotations

import json
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from src.data_access import get_requester_department, load_employees
from src.rules import detect_injection
from src.solution import handle_request
from src.ui_helpers import (
    badge_text,
    flag_color,
    format_telemetry,
    group_evidence,
    is_fallback,
)

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="AI Procurement Request Copilot", layout="wide")
st.title("AI Procurement Request Copilot")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")

dataset_source = st.sidebar.radio(
    "Request Dataset",
    ["Starter Requests (data/requests.json)", "Synthetic Edge Cases (evals/fixtures)"],
    index=0,
)

if dataset_source == "Synthetic Edge Cases (evals/fixtures)":
    fixtures_dir: str | None = str(ROOT / "evals" / "fixtures")
    requests_path = ROOT / "evals" / "fixtures" / "requests.json"
    st.sidebar.info("ℹ️ **Note:** Fixture vendors are loaded automatically by `run_local.py`.")
else:
    fixtures_dir = None
    requests_path = ROOT / "data" / "requests.json"

requests_data = json.loads(requests_path.read_text(encoding="utf-8"))
req_ids = [r["request_id"] for r in requests_data]
req_by_id = {r["request_id"]: r for r in requests_data}

selected_req_id = st.sidebar.selectbox(
    "Select Request",
    req_ids,
    format_func=lambda rid: f"{rid} - {req_by_id[rid].get('product_name') or 'Unnamed'}",
)
selected_request = req_by_id[selected_req_id]

architecture = st.sidebar.radio(
    "Architecture",
    ["single", "staged", "deterministic"],
    index=0,
)

st.sidebar.warning("⏱️ **Note:** LLM runs take about 20-30s. Deterministic runs are instant.")

run_copilot = st.sidebar.button("Run copilot", type="primary", use_container_width=True)

# --- SESSION STATE INITIALIZATION ---
if "decision_cache" not in st.session_state:
    st.session_state["decision_cache"] = {}
if "decision_log" not in st.session_state:
    st.session_state["decision_log"] = []

cache_key = f"{selected_req_id}_{architecture}_{fixtures_dir}"

if run_copilot:
    with st.spinner("Running copilot analysis..."):
        try:
            decision = handle_request(
                selected_request,
                architecture=architecture,
                fixtures_dir=fixtures_dir,
            )
            st.session_state["decision_cache"][cache_key] = {
                "decision": decision,
                "architecture": architecture,
            }
        except Exception as exc:
            st.error(f"Failed to execute procurement copilot: {exc}")

cached_entry = st.session_state["decision_cache"].get(cache_key)
current_decision = cached_entry["decision"] if cached_entry else None
cached_architecture = cached_entry.get("architecture", architecture) if cached_entry else architecture

# --- 3-PANEL LAYOUT ---
col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

# ==========================================
# PANEL 1: REQUEST DETAILS
# ==========================================
with col1:
    st.header("1. Request Details")

    dept = get_requester_department(selected_request, fixtures_dir=fixtures_dir)
    requester_id = selected_request.get("requester_id")
    requester_name = "Not specified"
    if requester_id:
        employees_df = load_employees(fixtures_dir=fixtures_dir)
        matches = employees_df[employees_df["employee_id"] == str(requester_id)]
        if not matches.empty:
            requester_name = str(matches.iloc[0]["name"])
            dept = dept or str(matches.iloc[0]["department"])
        else:
            requester_name = f"Employee {requester_id}"
    elif dept:
        requester_name = f"Department Member"

    cost_val = selected_request.get("annual_cost_usd")
    cost_display = f"${cost_val:,.2f}" if cost_val is not None else "N/A"
    integrations = selected_request.get("requested_integrations") or []
    integrations_display = ", ".join(integrations) if integrations else "None"

    st.markdown(f"**Request ID:** `{selected_request.get('request_id')}`")
    st.markdown(f"**Requester:** {requester_name} ({dept or 'Unknown'})")
    st.markdown(f"**Product:** {selected_request.get('product_name') or 'N/A'}")
    st.markdown(f"**Vendor:** {selected_request.get('vendor_name') or 'N/A'}")
    st.markdown(f"**Category:** {selected_request.get('category') or 'N/A'}")
    st.markdown(f"**Annual Cost:** {cost_display}")
    st.markdown(f"**User Count:** {selected_request.get('user_count') or 'N/A'}")
    st.markdown(f"**Data Access Level:** `{selected_request.get('data_access_level') or 'N/A'}`")
    st.markdown(f"**Integrations:** {integrations_display}")
    st.markdown(f"**Urgency:** {selected_request.get('urgency') or 'normal'}")

    st.markdown("---")
    st.markdown("**Business Justification:**")
    justification = selected_request.get("business_justification") or ""
    is_inj, _ = detect_injection(justification)
    if is_inj:
        st.markdown(
            '<span style="background-color: #fee2e2; color: #dc2626; border: 1px solid #f87171; '
            'padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold;">'
            '🚨 possible prompt injection</span>',
            unsafe_allow_html=True,
        )
    st.markdown(f"> {justification}" if justification else "> *None provided*")

# ==========================================
# PANEL 2: EVIDENCE
# ==========================================
with col2:
    st.header("2. Evidence")

    if current_decision is None:
        st.info("Click **Run copilot** in the sidebar to review evidence.")
    else:
        st.subheader("Risk Flags")
        if current_decision.risk_flags:
            badge_htmls = []
            for flag in current_decision.risk_flags:
                c = flag_color(flag)
                if c == "red":
                    bg, text_c, border_c = "#fee2e2", "#991b1b", "#f87171"
                elif c == "orange":
                    bg, text_c, border_c = "#ffedd5", "#9a3412", "#fb923c"
                else:
                    bg, text_c, border_c = "#f3f4f6", "#374151", "#d1d5db"
                badge_htmls.append(
                    f'<span style="background-color: {bg}; color: {text_c}; border: 1px solid {border_c}; '
                    f'padding: 3px 8px; border-radius: 5px; font-weight: 600; font-size: 0.85em; '
                    f'display: inline-block; margin: 2px 4px 2px 0;">{badge_text(flag)}</span>'
                )
            st.markdown(" ".join(badge_htmls), unsafe_allow_html=True)
        else:
            st.caption("No risk flags identified.")

        st.subheader("Missing Information")
        if current_decision.missing_information:
            for item in current_decision.missing_information:
                st.markdown(f"- ❓ {item}")
        else:
            st.caption("No missing information.")

        st.subheader("Collected Evidence")
        grouped = group_evidence(current_decision.evidence)
        if grouped:
            for source, items in grouped.items():
                with st.expander(f"📁 {source.replace('_', ' ').title()} ({len(items)})", expanded=True):
                    for it in items:
                        finding = it.get("finding", "")
                        reference = it.get("reference")
                        ref_str = f" *(Ref: `{reference}`)*" if reference else ""
                        st.markdown(f"- {finding}{ref_str}")
        else:
            st.caption("No evidence items found.")

# ==========================================
# PANEL 3: RECOMMENDATION + ACTION
# ==========================================
with col3:
    st.header("3. Recommendation")

    if current_decision is None:
        st.info("Click **Run copilot** in the sidebar to produce a recommendation.")
    else:
        if is_fallback(current_decision):
            st.warning("⚠️ LLM unavailable: deterministic fallback shown")

        st.info("🛡️ **Advisory only: a human must decide**")

        st.subheader("Recommendation")
        cat = current_decision.recommendation_category
        cat_str = cat.value if hasattr(cat, "value") else str(cat or "N/A")
        st.markdown(
            f'<span style="background-color: #ede9fe; color: #5b21b6; border: 1px solid #c4b5fd; '
            f'padding: 3px 8px; border-radius: 5px; font-weight: 600; font-size: 0.85em; '
            f'display: inline-block; margin-bottom: 6px;">🏷️ {badge_text(cat_str)}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**{current_decision.recommendation}**")

        st.subheader("Required Approvals")
        if current_decision.required_approvals:
            for idx, apprv in enumerate(current_decision.required_approvals, 1):
                st.markdown(f"**{idx}.** ☐ {apprv}")
        else:
            st.caption("No approvals required.")

        st.subheader("Next Step")
        st.markdown(f"👉 *{current_decision.next_step}*")

        st.markdown("---")
        st.subheader("Reviewer Action")
        comment = st.text_input("Reviewer comment (optional)", key=f"comment_{selected_req_id}")

        def record_action(act: str) -> None:
            st.session_state["decision_log"].insert(
                0,
                {
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Request ID": selected_req_id,
                    "Action": act,
                    "Comment": comment.strip() if comment else "-",
                    "Recommendation": current_decision.recommendation,
                },
            )

        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            if st.button("Approve", type="primary", use_container_width=True):
                record_action("Approve")
                st.success("Decision logged: Approved")
        with bcol2:
            if st.button("Request more info", use_container_width=True):
                record_action("Request more info")
                st.info("Decision logged: Request more info")
        with bcol3:
            if st.button("Escalate", use_container_width=True):
                record_action("Escalate")
                st.warning("Decision logged: Escalate")

        st.caption("These record the human reviewer's choice only. Nothing is sent to any system.")

        st.subheader("Decision Log")
        if st.session_state.get("decision_log"):
            st.dataframe(pd.DataFrame(st.session_state["decision_log"]), use_container_width=True)
        else:
            st.caption("No human decisions logged yet.")

        st.markdown("---")
        st.subheader("Telemetry")
        tel = format_telemetry(current_decision)
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.metric("Architecture", cached_architecture)
            st.metric("LLM Calls", tel["llm_calls"])
            st.metric("Latency", f"{tel['latency_ms']:.1f} ms" if tel.get("latency_ms") else "N/A")
        with tcol2:
            st.metric("Model", tel["model"])
            st.metric("Tool Calls", tel["tool_calls"])
            st.metric("Guardrail Corrections", tel["guardrail_corrections"])

        if tel.get("tool_names"):
            st.caption(f"**Tools executed:** {', '.join(tel['tool_names'])}")
