from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from src.agent_common import (
    DraftDecision,
    SINGLE_AGENT_SYSTEM,
    build_untrusted_block,
    prefetch_vendor,
)
from src.contracts import ProcurementDecision
from src.guardrails import apply_guardrails
from src.llm import LLMClient, LLMUnavailableError, get_llm_client
from src.tools import bind_tools, gemini_function_declarations


def run_single_agent(
    request: dict,
    fixtures_dir: Path | str | None = None,
    client: LLMClient | None = None,
    use_cache: bool | None = None,
    vendor_api_result: dict | None = None,
    started_at: float | None = None,
) -> ProcurementDecision:
    """Run Single Agent (Architecture A).

    1. Run tool loop with bound tools to investigate budget, catalog, vendor risk, and policy floor.
    2. Generate structured DraftDecision adhering to Pydantic schema.
    3. Apply deterministic safety & policy guardrails.
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

    # 1. Tool loop turn(s)
    try:
        loop_res = llm_client.run_tool_loop(
            system=SINGLE_AGENT_SYSTEM,
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

    # 2. Structured decision generation
    draft_user_prompt = (
        f"{untrusted_request_block}\n\n"
        f"<tool_analysis_summary>\n{loop_res.text}\n</tool_analysis_summary>\n\n"
        "Based on the tool analysis and untrusted request data, synthesize the final DraftDecision."
    )

    try:
        draft_res = llm_client.generate_structured(
            system=SINGLE_AGENT_SYSTEM,
            user=draft_user_prompt,
            schema=DraftDecision,
            use_cache=use_cache,
        )
        llm_calls += draft_res.llm_calls
        draft_dict = draft_res.parsed
    except LLMUnavailableError:
        draft_dict = None

    # 3. Guardrails
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
