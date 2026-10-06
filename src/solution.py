from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from src.agent_common import prefetch_vendor
from src.contracts import Architecture, ProcurementDecision
from src.data_access import get_request
from src.guardrails import apply_guardrails
from src.single_agent import run_single_agent
from src.staged_agents import run_staged_agents


def handle_request(
    request_or_id: dict | str,
    architecture: Architecture | str = "single",
    fixtures_dir: Path | str | None = None,
    use_cache: bool | None = None,
    client: Any = None,
    vendor_api_result: dict | None = None,
) -> ProcurementDecision:
    """Assessment entrypoint adapter.

    Executes procurement evaluation under 'single', 'staged', or 'deterministic' architecture.
    """
    started_at = time.time()

    if isinstance(request_or_id, str):
        request = get_request(request_or_id, fixtures_dir=fixtures_dir)
    elif isinstance(request_or_id, dict):
        request = request_or_id
    else:
        raise ValueError(f"Invalid request input type: {type(request_or_id)}")

    if architecture == "deterministic":
        if vendor_api_result is None:
            vendor_api_result = prefetch_vendor(request, fixtures_dir=fixtures_dir)
        vendor_name = request.get("vendor_name")
        has_vendor = bool(vendor_name and str(vendor_name).strip() and str(vendor_name).strip().lower() != "unknown")
        det_tool_calls = 1 if has_vendor else 0
        det_tool_names = ["get_vendor_risk_prefetch"] if has_vendor else []
        return apply_guardrails(
            draft=None,
            request=request,
            vendor_api_result=vendor_api_result,
            started_at=started_at,
            llm_calls=0,
            tool_calls=det_tool_calls,
            tool_names=det_tool_names,
            model_name="deterministic",
            fixtures_dir=fixtures_dir,
        )

    if architecture == "single":
        return run_single_agent(
            request=request,
            fixtures_dir=fixtures_dir,
            client=client,
            use_cache=use_cache,
            vendor_api_result=vendor_api_result,
            started_at=started_at,
        )

    if architecture == "staged":
        return run_staged_agents(
            request=request,
            fixtures_dir=fixtures_dir,
            client=client,
            use_cache=use_cache,
            vendor_api_result=vendor_api_result,
            started_at=started_at,
        )

    raise ValueError(f"Unknown architecture: '{architecture}'. Allowed architectures: 'single', 'staged', 'deterministic'")
