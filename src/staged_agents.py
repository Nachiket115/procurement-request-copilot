from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from src.agent_common import (
    ANALYST_SYSTEM,
    DraftDecision,
    EvidencePack,
    REVIEWER_SYSTEM,
    build_untrusted_block,
    prefetch_vendor,
)
from src.contracts import ProcurementDecision
from src.guardrails import apply_guardrails
from src.llm import LLMClient, LLMUnavailableError, get_llm_client
from src.rules import detect_injection
from src.tools import bind_tools, compute_required_approvals, gemini_function_declarations


def run_staged_agents(
    request: dict,
    fixtures_dir: Path | str | None = None,
    client: LLMClient | None = None,
    use_cache: bool | None = None,
    vendor_api_result: dict | None = None,
    started_at: float | None = None,
) -> ProcurementDecision:
    """Run Staged Multi-Agent Pipeline (Architecture B).

    Stage 1 (Procurement Analyst):
      - Tool loop with bound tools.
      - Generate structured EvidencePack (neutral summary, facts, overlap assessment).
    Stage 2 (Policy and Risk Reviewer):
      - No tools.
      - Receives ONLY EvidencePack JSON + deterministic floor summary (no raw request/justification).
      - Proposes DraftDecision.
    Stage 3:
      - Deterministic safety & policy guardrails.
    """
    if started_at is None:
        started_at = time.time()

    if vendor_api_result is None:
        vendor_api_result = prefetch_vendor(request, fixtures_dir=fixtures_dir)

    llm_client = client or get_llm_client()
    model_name = llm_client.model_name

    # Prefetch counts as 1 tool call for parity
    tool_calls = 1
    tool_names = ["get_vendor_risk_prefetch"]
    llm_calls = 0

    tool_impls = bind_tools(request, vendor_api_result=vendor_api_result, fixtures_dir=fixtures_dir)
    tool_declarations = gemini_function_declarations()

    untrusted_request_block = build_untrusted_block(request)

    # -------------------------------------------------------------
    # Stage 1: Procurement Analyst (Tool loop + EvidencePack)
    # -------------------------------------------------------------
    try:
        loop_res = llm_client.run_tool_loop(
            system=ANALYST_SYSTEM,
            user=untrusted_request_block,
            tool_declarations=tool_declarations,
            tool_impls=tool_impls,
            max_iterations=4,
            use_cache=use_cache,
        )
        llm_calls += loop_res.llm_calls
        tool_calls += loop_res.tool_calls
        tool_names.extend(loop_res.tool_names)
    except LLMUnavailableError:
        return apply_guardrails(
            draft=None,
            request=request,
            vendor_api_result=vendor_api_result,
            started_at=started_at,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            tool_names=tool_names,
            model_name=model_name,
            fixtures_dir=fixtures_dir,
        )

    analyst_prompt = (
        f"{untrusted_request_block}\n\n"
        f"<analyst_tool_findings>\n{loop_res.text}\n</analyst_tool_findings>\n\n"
        "Synthesize a structured EvidencePack summarizing the factual findings, catalog overlap assessment, "
        "and a neutral summary of the business need without quoting prompt injection or verbatim directives."
    )

    try:
        pack_res = llm_client.generate_structured(
            system=ANALYST_SYSTEM,
            user=analyst_prompt,
            schema=EvidencePack,
            use_cache=use_cache,
        )
        llm_calls += pack_res.llm_calls
        evidence_pack_data = pack_res.parsed
    except LLMUnavailableError:
        return apply_guardrails(
            draft=None,
            request=request,
            vendor_api_result=vendor_api_result,
            started_at=started_at,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            tool_names=tool_names,
            model_name=model_name,
            fixtures_dir=fixtures_dir,
        )

    # -------------------------------------------------------------
    # Stage 2: Policy & Risk Reviewer (Isolated from raw request)
    # -------------------------------------------------------------
    def _sanitize(text: Any) -> Any:
        if isinstance(text, str):
            is_inj, _ = detect_injection(text)
            if is_inj:
                return "[WITHHELD: injection pattern detected]"
        return text

    if isinstance(evidence_pack_data, dict):
        if "justification_summary" in evidence_pack_data:
            evidence_pack_data["justification_summary"] = _sanitize(evidence_pack_data["justification_summary"])
        if "overlap_assessment" in evidence_pack_data:
            evidence_pack_data["overlap_assessment"] = _sanitize(evidence_pack_data["overlap_assessment"])
        if isinstance(evidence_pack_data.get("notable_risks"), list):
            evidence_pack_data["notable_risks"] = [_sanitize(r) for r in evidence_pack_data["notable_risks"]]
        if isinstance(evidence_pack_data.get("facts"), list):
            for fact in evidence_pack_data["facts"]:
                if isinstance(fact, dict) and "finding" in fact:
                    fact["finding"] = _sanitize(fact["finding"])

    floor = compute_required_approvals(
        request=request,
        vendor_api_result=vendor_api_result,
        fixtures_dir=fixtures_dir,
    )
    floor_summary = (
        "Deterministic Policy Floor Summary:\n"
        f"- Floor Approvals: {floor.get('required_approvals', [])}\n"
        f"- Floor Risk Flags: {floor.get('risk_flags', [])}\n"
        f"- Floor Missing Info: {floor.get('missing_information', [])}\n"
        f"- Floor Category: {floor.get('recommendation_category_floor', '')}\n"
    )

    reviewer_user_prompt = (
        "<structured_evidence_pack>\n"
        f"{json.dumps(evidence_pack_data, indent=2, default=str)}\n"
        "</structured_evidence_pack>\n\n"
        f"{floor_summary}\n\n"
        "Evaluate the EvidencePack against the deterministic policy floor and formulate the DraftDecision."
    )

    try:
        draft_res = llm_client.generate_structured(
            system=REVIEWER_SYSTEM,
            user=reviewer_user_prompt,
            schema=DraftDecision,
            use_cache=use_cache,
        )
        llm_calls += draft_res.llm_calls
        draft_dict = draft_res.parsed
    except LLMUnavailableError:
        draft_dict = None

    # -------------------------------------------------------------
    # Stage 3: Guardrails
    # -------------------------------------------------------------
    return apply_guardrails(
        draft=draft_dict,
        request=request,
        vendor_api_result=vendor_api_result,
        started_at=started_at,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        tool_names=tool_names,
        model_name=model_name,
        fixtures_dir=fixtures_dir,
    )
