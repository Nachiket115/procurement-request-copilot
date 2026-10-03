from __future__ import annotations

import copy
from decimal import Decimal
import logging
from pathlib import Path
from typing import Any, Callable
import pandas as pd

from src.data_access import (
    get_department_budget,
    get_requester_department,
    load_software_catalog,
    load_vendors,
)
from src.rules import (
    check_budget,
    detect_injection,
    evaluate_request,
    find_catalog_overlap,
    reconcile_vendor,
)
from src.vendor_client import get_vendor_risk

logger = logging.getLogger(__name__)


def check_department_budget(
    department: str,
    annual_cost_usd: float | int | str | Decimal | None = None,
    fixtures_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Check department software budget against requested annual cost.

    When fixtures_dir is None, the existing FIXTURES_DIR env fallback in data_access still applies.
    """
    try:
        b_res = get_department_budget(
            str(department) if department is not None else "",
            fixtures_dir=fixtures_dir,
        )
        chk = check_budget(annual_cost_usd, b_res)
        result: dict[str, Any] = {
            "status": "ok",
            "department": b_res.get("department", department),
            "found": chk.get("found", False),
            "budget_insufficient": chk.get("budget_insufficient", False),
            "available_usd": float(chk["available_usd"]) if "available_usd" in chk else b_res.get("available_usd"),
            "annual_software_budget_usd": b_res.get("annual_software_budget_usd"),
            "committed_usd": b_res.get("committed_usd"),
        }
        if "cost_usd" in chk:
            result["cost_usd"] = float(chk["cost_usd"])
        if "missing_department" in chk:
            result["missing_department"] = chk["missing_department"]
        if "error" in chk:
            result["error"] = chk["error"]
        return result
    except Exception as e:
        logger.exception("Error in check_department_budget: %s", e)
        return {"status": "unavailable", "error": str(e)}


def check_software_catalog(
    product_name: str | None = None,
    vendor_name: str | None = None,
    category: str | None = None,
    fixtures_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Check software catalog for existing tools matching product, vendor, or category.

    When fixtures_dir is None, the existing FIXTURES_DIR env fallback in data_access still applies.
    """
    try:
        req = {
            "product_name": product_name,
            "vendor_name": vendor_name,
            "category": category,
        }
        catalog_df = load_software_catalog(fixtures_dir=fixtures_dir)
        raw_matches = find_catalog_overlap(req, catalog_df=catalog_df)
        matches: list[dict[str, Any]] = []
        for m in raw_matches:
            matches.append({
                "software_id": str(m.get("software_id", "")),
                "product_name": str(m.get("product_name", "")),
                "vendor_name": str(m.get("vendor_name", "")),
                "category": str(m.get("category", "")),
                "scope": str(m.get("scope", "")),
                "licensed_seats": int(m["licensed_seats"]) if pd.notna(m.get("licensed_seats")) else None,
                "annual_cost_usd": float(m["annual_cost_usd"]) if pd.notna(m.get("annual_cost_usd")) else None,
                "status": str(m.get("status", "")),
                "overlap_type": str(m.get("overlap_type", "")),
            })
        return {
            "status": "ok",
            "matches_found": len(matches),
            "matches": matches,
        }
    except Exception as e:
        logger.exception("Error in check_software_catalog: %s", e)
        return {"status": "unavailable", "error": str(e)}


def verify_vendor_risk(
    vendor_name: str,
    fixtures_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Verify vendor risk against external API and reconcile with internal vendor registry.

    When fixtures_dir is None, the existing FIXTURES_DIR env fallback in data_access still applies.
    """
    try:
        if not vendor_name or not str(vendor_name).strip():
            return {"status": "unavailable", "error": "Missing vendor_name"}
        name = str(vendor_name).strip()
        api_res = get_vendor_risk(name)

        # Sanitize notes if prompt injection detected
        api_notes = api_res.get("notes")
        if api_notes:
            is_inj, _ = detect_injection(api_notes)
            if is_inj:
                api_res = dict(api_res)
                api_res["notes"] = "[WITHHELD: injection pattern detected]"

        vendors_df = load_vendors(fixtures_dir=fixtures_dir)
        registry_row: dict[str, Any] | None = None
        if not vendors_df.empty:
            clean_name = name.lower()
            matches = vendors_df[vendors_df["vendor_name"].astype(str).str.strip().str.lower() == clean_name]
            if not matches.empty:
                raw_row = matches.iloc[0].to_dict()
                registry_row = {k: (None if pd.isna(v) else v) for k, v in raw_row.items()}
                reg_notes = registry_row.get("notes")
                if reg_notes:
                    is_inj_reg, _ = detect_injection(reg_notes)
                    if is_inj_reg:
                        registry_row["notes"] = "[WITHHELD: injection pattern detected]"

        rec = reconcile_vendor(registry_row, api_res)
        return {
            "status": "ok",
            "vendor_name": name,
            "api_result": api_res,
            "registry_row": registry_row,
            "flags": rec.get("flags", []),
            "evidence": rec.get("evidence", []),
            "is_expired": rec.get("is_expired", False),
        }
    except Exception as e:
        logger.exception("Error in verify_vendor_risk: %s", e)
        return {"status": "unavailable", "error": str(e)}


def compute_required_approvals(
    request: dict,
    vendor_api_result: dict | None = None,
    fixtures_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Deterministically compute policy floor approvals, risk flags, missing info, and category floor.

    When fixtures_dir is None, the existing FIXTURES_DIR env fallback in data_access still applies.
    """
    try:
        if not isinstance(request, dict):
            return {"status": "unavailable", "error": f"Invalid request type: {type(request)}"}
        eval_res = evaluate_request(
            request,
            vendor_api_result=vendor_api_result,
            fixtures_dir=fixtures_dir,
        )
        return {
            "status": "ok",
            "required_approvals": list(eval_res.get("required_approvals", [])),
            "risk_flags": list(eval_res.get("risk_flags", [])),
            "missing_information": list(eval_res.get("missing_information", [])),
            "recommendation_category_floor": str(eval_res.get("recommendation_category_floor", "")),
            "recommendation_category": str(eval_res.get("recommendation_category_floor", "")),
            "evidence": list(eval_res.get("evidence", [])),
            # Always True by policy Section 11, hardcoded here rather than read from rules output.
            "human_review_required": True,
        }
    except Exception as e:
        logger.exception("Error in compute_required_approvals: %s", e)
        return {"status": "unavailable", "error": str(e)}


def bind_tools(
    request: dict,
    vendor_api_result: dict | None = None,
    fixtures_dir: Path | str | None = None,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Bind tools to the real request facts and vendor risk result via closure.

    The model must not be able to alter the facts the deterministic checks run on.
    Therefore, compute_required_approvals and check_department_budget are bound to
    the actual request data and take no arguments from the LLM.
    """
    dept = get_requester_department(request, fixtures_dir=fixtures_dir) or ""
    return {
        "check_department_budget": lambda **kwargs: check_department_budget(
            department=dept,
            annual_cost_usd=request.get("annual_cost_usd"),
            fixtures_dir=fixtures_dir,
        ),
        "check_software_catalog": lambda **kwargs: check_software_catalog(
            product_name=kwargs.get("product_name"),
            vendor_name=kwargs.get("vendor_name"),
            category=kwargs.get("category"),
            fixtures_dir=fixtures_dir,
        ),
        "verify_vendor_risk": lambda **kwargs: verify_vendor_risk(
            vendor_name=kwargs.get("vendor_name", request.get("vendor_name", "")),
            fixtures_dir=fixtures_dir,
        ),
        "compute_required_approvals": lambda **kwargs: compute_required_approvals(
            request=request,
            vendor_api_result=vendor_api_result,
            fixtures_dir=fixtures_dir,
        ),
    }


TOOL_REGISTRY = {
    "check_department_budget": check_department_budget,
    "check_software_catalog": check_software_catalog,
    "verify_vendor_risk": verify_vendor_risk,
    "compute_required_approvals": compute_required_approvals,
}

# TOOL_SCHEMAS for Gemini / LLM function calling.
# Notice: compute_required_approvals and check_department_budget take empty properties
# because the model must not be able to alter the facts the deterministic checks run on.
TOOL_SCHEMAS = [
    {
        "name": "check_department_budget",
        "description": "Check department available software budget against requested annual cost for the active procurement request.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "check_software_catalog",
        "description": "Check software catalog for existing tools matching product name, vendor, or category.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Product or software tool name.",
                },
                "vendor_name": {
                    "type": "string",
                    "description": "Vendor or publisher name.",
                },
                "category": {
                    "type": "string",
                    "description": "Software category or use-case domain.",
                },
            },
        },
    },
    {
        "name": "verify_vendor_risk",
        "description": "Verify external vendor risk and reconcile with internal vendor registry.",
        "parameters": {
            "type": "object",
            "properties": {
                "vendor_name": {
                    "type": "string",
                    "description": "Name of the vendor to verify.",
                },
            },
            "required": ["vendor_name"],
        },
    },
    {
        "name": "compute_required_approvals",
        "description": "Deterministically compute policy floor approvals, risk flags, missing information, and category floor for the active procurement request.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


def gemini_function_declarations() -> list[dict]:
    """Return function declarations formatted for Gemini API.

    Gemini rejects function declarations with empty properties schemas, so the
    'parameters' key is omitted for any tool with an empty properties dict.
    """
    declarations = []
    for schema in TOOL_SCHEMAS:
        decl = copy.deepcopy(schema)
        params = decl.get("parameters", {})
        if isinstance(params, dict) and not params.get("properties"):
            decl.pop("parameters", None)
        declarations.append(decl)
    return declarations
