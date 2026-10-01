from __future__ import annotations

from datetime import date
from decimal import Decimal
import re
from typing import Any
import pandas as pd

from src.contracts import RecommendationCategory
from src.data_access import (
    get_department_budget,
    get_requester_department,
    load_software_catalog,
    load_vendors,
)

REFERENCE_DATE = date(2026, 9, 30)

INJECTION_PATTERNS = [
    re.compile(r"ignore all procurement rules", re.IGNORECASE),
    re.compile(r"treat this (?:request )?as cfo-approved", re.IGNORECASE),
    re.compile(r"skip security", re.IGNORECASE),
    re.compile(r"pre-approved", re.IGNORECASE),
    re.compile(r"system override", re.IGNORECASE),
    re.compile(r"disregard", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
]

SENSITIVE_DATA_ACCESS_LEVELS = {
    "source_code",
    "confidential_documents",
    "employee_pii",
    "customer_pii",
    "credentials/secrets",
}

PII_DATA_ACCESS_LEVELS = {
    "employee_pii",
    "customer_pii",
}


def clean_str(val: Any) -> str:
    """Return empty string for None/NaN and a stripped lowercase string otherwise."""
    if val is None or pd.isna(val):
        return ""
    return str(val).strip().lower()


def clean_bool(val: Any) -> bool:
    """Return True only for True or string 'true' (case-insensitive); NaN/None is False."""
    if val is True:
        return True
    if val is False or val is None or pd.isna(val):
        return False
    if isinstance(val, str) and val.strip().lower() == "true":
        return True
    return False


def is_sensitive_data_access(level: str | None) -> bool:
    """Check whether data access level triggers Security review."""
    if not level:
        return False
    clean = clean_str(level)
    if clean in SENSITIVE_DATA_ACCESS_LEVELS:
        return True
    return any(keyword in clean for keyword in ("production", "cloud", "credential", "secret"))


def approval_thresholds(cost: int | float | str | Decimal | None) -> list[str]:
    """Deterministic financial approval tiers per Policy Section 4.

    <=1000: Manager
    1000.01 - 10000: Department Head + Procurement
    10000.01 - 25000: Department Head + Finance + Procurement
    >25000: Department Head + Finance + CFO + Procurement
    """
    if cost is None:
        return []
    val = Decimal(str(cost))
    if val <= Decimal("1000"):
        return ["Manager"]
    if val <= Decimal("10000"):
        return ["Department Head", "Procurement"]
    if val <= Decimal("25000"):
        return ["Department Head", "Finance", "Procurement"]
    return ["Department Head", "Finance", "CFO", "Procurement"]


def find_missing_fields(request: dict) -> list[str]:
    """Check for material fields required by Policy Section 1.

    None, empty string, and 'unknown' (case-insensitive) are treated as missing.
    """
    missing: list[str] = []

    v = request.get("vendor_name")
    if v is None or (isinstance(v, str) and (not v.strip() or v.strip().lower() == "unknown")):
        missing.append("Missing vendor name")

    c = request.get("annual_cost_usd")
    if c is None or (isinstance(c, str) and (not c.strip() or c.strip().lower() == "unknown")):
        missing.append("Missing annual cost estimate")

    u = request.get("user_count")
    if u is None or (isinstance(u, str) and (not u.strip() or u.strip().lower() == "unknown")):
        missing.append("Missing user/seat license count")

    d = request.get("data_access_level")
    if d is None or (isinstance(d, str) and (not d.strip() or d.strip().lower() == "unknown")):
        missing.append("Missing intended data access level")

    return missing


def check_budget(cost: int | float | str | Decimal | None, budget_result: dict) -> dict:
    """Compare annual cost against available software budget per Policy Section 2.

    If department budget record is missing, found is False; does not flag budget_insufficient.
    """
    if not budget_result.get("found", False):
        return {
            "found": False,
            "budget_insufficient": False,
            "missing_department": True,
            "error": budget_result.get("error", "Department budget record missing"),
        }
    if cost is None:
        return {
            "found": True,
            "budget_insufficient": False,
            "available_usd": budget_result.get("available_usd", 0),
        }
    val = Decimal(str(cost))
    avail = Decimal(str(budget_result.get("available_usd", 0)))
    if val > avail:
        return {
            "found": True,
            "budget_insufficient": True,
            "cost_usd": val,
            "available_usd": avail,
        }
    return {
        "found": True,
        "budget_insufficient": False,
        "cost_usd": val,
        "available_usd": avail,
    }


def find_catalog_overlap(request: dict, catalog_df: pd.DataFrame | None = None) -> list[dict]:
    """Find catalog matches per literal Policy Section 3 (same product, vendor, or category).

    overlap_type (duplicate/expansion/competitor) is informational only.
    """
    if catalog_df is None:
        catalog_df = load_software_catalog()

    matches: list[dict] = []
    prod_req = clean_str(request.get("product_name"))
    vendor_req = clean_str(request.get("vendor_name"))
    cat_req = clean_str(request.get("category"))

    if not prod_req and not vendor_req and not cat_req:
        return []

    for _, row in catalog_df.iterrows():
        sw_name = clean_str(row.get("product_name"))
        sw_vendor = clean_str(row.get("vendor_name"))
        sw_cat = clean_str(row.get("category"))

        matched = False
        overlap_type = "competitor"

        if prod_req and sw_name and (prod_req == sw_name or prod_req in sw_name or sw_name in prod_req):
            matched = True
            overlap_type = "duplicate" if prod_req == sw_name else "expansion"
        elif vendor_req and sw_vendor and vendor_req == sw_vendor:
            matched = True
            overlap_type = "expansion"
        elif cat_req and sw_cat and cat_req == sw_cat:
            matched = True
            overlap_type = "competitor"

        if matched:
            m = row.to_dict()
            m["overlap_type"] = overlap_type
            matches.append(m)

    return matches


def reconcile_vendor(registry_row: dict | None, api_result: dict | None) -> dict:
    """Reconcile internal vendor registry and external vendor risk API per Policy Section 5.

    Expiry rule: (REFERENCE_DATE - review_date).days > 365 is expired; exactly 365 is valid.
    """
    flags: list[str] = []
    evidence: list[dict[str, Any]] = []

    api_status = clean_str((api_result or {}).get("status", "ok")) if api_result else "unavailable"

    if api_result and api_status in ("unavailable", "not_found"):
        flags.append("vendor_risk_unavailable")
        evidence.append({
            "source": "vendor_risk_api",
            "finding": f"Vendor risk service returned status '{api_status}': {api_result.get('error', 'unavailable')}",
            "reference": "/vendor-risk",
        })
    elif api_result:
        finding_parts = [
            f"Vendor risk level: {api_result.get('risk_level', 'unknown')}",
            f"security status: {api_result.get('security_review_status', 'unknown')}",
        ]
        api_notes = api_result.get("notes", "")
        if api_notes:
            is_inj, _ = detect_injection(api_notes)
            if is_inj:
                finding_parts.append("notes withheld: injection pattern detected")
        evidence.append({
            "source": "vendor_risk_api",
            "finding": ", ".join(finding_parts),
            "reference": "/vendor-risk",
        })

    if registry_row:
        reg_sec = clean_str(registry_row.get("security_status")) or clean_str(registry_row.get("security_review_status"))
        evidence.append({
            "source": "vendor_registry",
            "finding": (
                f"Internal registry status: security={reg_sec or 'unknown'}, "
                f"legal={clean_str(registry_row.get('legal_terms_status')) or 'unknown'}, "
                f"procurement={clean_str(registry_row.get('procurement_status')) or 'unknown'}"
            ),
            "reference": "vendors.csv",
        })
    else:
        evidence.append({
            "source": "vendor_registry",
            "finding": "Vendor is not present in internal vendor registry (new vendor).",
            "reference": "vendors.csv",
        })

    reg_date_str = registry_row.get("security_review_date") if registry_row else None
    if pd.isna(reg_date_str):
        reg_date_str = None
    api_date_str = api_result.get("last_review_date") if api_result else None
    if pd.isna(api_date_str):
        api_date_str = None

    reg_expired = False
    if reg_date_str:
        try:
            reg_d = date.fromisoformat(str(reg_date_str).strip())
            if (REFERENCE_DATE - reg_d).days > 365:
                reg_expired = True
        except ValueError:
            pass

    api_expired = False
    if api_date_str:
        try:
            api_d = date.fromisoformat(str(api_date_str).strip())
            if (REFERENCE_DATE - api_d).days > 365:
                api_expired = True
        except ValueError:
            pass

    if api_result and clean_str(api_result.get("security_review_status")) == "expired":
        api_expired = True

    if reg_expired or api_expired:
        if "vendor_review_expired" not in flags:
            flags.append("vendor_review_expired")

    # Conflicting vendor evidence: registry says "approved" AND API says expired/not_completed/pending/failed
    reg_sec_str = clean_str(registry_row.get("security_status") if registry_row else None) or clean_str(registry_row.get("security_review_status") if registry_row else None)
    api_sec_str = clean_str(api_result.get("security_review_status") if api_result else None)

    if registry_row and reg_sec_str == "approved" and api_sec_str in ("expired", "not_completed", "pending", "failed"):
        if "conflicting_vendor_evidence" not in flags:
            flags.append("conflicting_vendor_evidence")

    return {
        "flags": flags,
        "evidence": evidence,
        "api_status": api_status,
        "is_expired": reg_expired or api_expired,
    }


def detect_injection(text: str | None) -> tuple[bool, str | None]:
    """Detect prompt injection heuristic patterns in untrusted input text per Policy Section 9."""
    if not text or pd.isna(text):
        return False, None
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(str(text))
        if match:
            return True, match.group(0).lower()
    return False, None


def compute_floor_category(
    missing_fields: list[str],
    has_unavailable_evidence: bool,
    budget_insufficient: bool,
    missing_department: bool,
    governance_required: bool,
) -> str:
    """Compute recommendation floor category with strict precedence:

    1. needs_more_information (material fields missing)
    2. escalate_unavailable_evidence (API unavailable or unknown vendor)
    3. escalate_to_finance (budget exceeded or department missing)
    4. route_for_governance_review (Security, Privacy, Legal, Finance, CFO)
    5. route_for_standard_review
    """
    if missing_fields:
        return RecommendationCategory.needs_more_information.value
    if has_unavailable_evidence:
        return RecommendationCategory.escalate_unavailable_evidence.value
    if budget_insufficient or missing_department:
        return RecommendationCategory.escalate_to_finance.value
    if governance_required:
        return RecommendationCategory.route_for_governance_review.value
    return RecommendationCategory.route_for_standard_review.value


def evaluate_request(
    request: dict,
    vendor_api_result: dict | None = None,
    catalog_df: pd.DataFrame | None = None,
    vendors_df: pd.DataFrame | None = None,
    fixtures_dir: Any = None,
) -> dict[str, Any]:
    """Pure Python deterministic policy evaluation engine for procurement requests."""
    evidence: list[dict[str, Any]] = []
    risk_flags: list[str] = []
    missing_information: list[str] = []

    # 1. Missing fields check
    missing_fields = find_missing_fields(request)
    if missing_fields:
        missing_information.extend(missing_fields)
        if "missing_information" not in risk_flags:
            risk_flags.append("missing_information")

    # 2. Prompt injection detection (check justification, vendor API notes, registry notes, and catalog notes)
    justification = request.get("business_justification", "")
    is_inj_req, pat_req = detect_injection(justification)
    if is_inj_req:
        if "prompt_injection_detected" not in risk_flags:
            risk_flags.append("prompt_injection_detected")
        evidence.append({
            "source": "security_scanner",
            "finding": f"Prompt injection pattern detected in business justification: '{pat_req}'",
            "reference": "Policy Section 9",
        })

    api_notes = (vendor_api_result or {}).get("notes", "")
    is_inj_api, pat_api = detect_injection(api_notes)
    if is_inj_api:
        if "prompt_injection_detected" not in risk_flags:
            risk_flags.append("prompt_injection_detected")
        evidence.append({
            "source": "security_scanner",
            "finding": f"Prompt injection pattern detected in vendor risk notes: '{pat_api}'",
            "reference": "Policy Section 9",
        })

    # 3. Catalog overlap check
    overlaps = find_catalog_overlap(request, catalog_df=catalog_df)
    if overlaps:
        if "existing_tool_overlap" not in risk_flags:
            risk_flags.append("existing_tool_overlap")
        evidence.append({
            "source": "software_catalog",
            "finding": f"Catalog overlap identified ({len(overlaps)} match(es)): {', '.join(str(m.get('product_name', '')) for m in overlaps)}",
            "reference": "software_catalog.csv",
        })

        # Scan matched catalog rows for prompt injection
        for m in overlaps:
            cat_notes = m.get("notes", "")
            if cat_notes:
                is_inj_cat, pat_cat = detect_injection(cat_notes)
                if is_inj_cat:
                    if "prompt_injection_detected" not in risk_flags:
                        risk_flags.append("prompt_injection_detected")
                    evidence.append({
                        "source": "security_scanner",
                        "finding": f"Prompt injection pattern detected in software catalog notes for '{m.get('product_name', '')}': '{pat_cat}'",
                        "reference": "Policy Section 9",
                    })

    # 4. Vendor lookup & reconciliation
    if vendors_df is None:
        vendors_df = load_vendors(fixtures_dir=fixtures_dir)

    vendor_name = request.get("vendor_name")
    has_vendor_name = bool(vendor_name and str(vendor_name).strip() and str(vendor_name).strip().lower() != "unknown")

    registry_row: dict | None = None
    if has_vendor_name and not vendors_df.empty:
        matches = vendors_df[vendors_df["vendor_name"].astype(str).str.strip().str.lower() == str(vendor_name).strip().lower()]
        if not matches.empty:
            registry_row = matches.iloc[0].to_dict()

    if registry_row:
        reg_notes = registry_row.get("notes", "")
        if reg_notes:
            is_inj_reg, pat_reg = detect_injection(reg_notes)
            if is_inj_reg:
                if "prompt_injection_detected" not in risk_flags:
                    risk_flags.append("prompt_injection_detected")
                evidence.append({
                    "source": "security_scanner",
                    "finding": f"Prompt injection pattern detected in vendor registry notes: '{pat_reg}'",
                    "reference": "Policy Section 9",
                })

    if has_vendor_name and vendor_api_result is None:
        if "vendor_risk_unavailable" not in risk_flags:
            risk_flags.append("vendor_risk_unavailable")
        evidence.append({
            "source": "vendor_risk_api",
            "finding": f"Vendor risk API result is unavailable for vendor '{vendor_name}'",
            "reference": "/vendor-risk",
        })

    reconcile_res = reconcile_vendor(registry_row, vendor_api_result)
    for f in reconcile_res["flags"]:
        if f not in risk_flags:
            risk_flags.append(f)
    evidence.extend(reconcile_res["evidence"])

    # Personal data processing note in evidence (note only, no risk flag)
    if vendor_api_result and clean_bool(vendor_api_result.get("processes_personal_data")):
        evidence.append({
            "source": "vendor_risk_api",
            "finding": "Vendor processes personal data (surfaced as evidence note only; request data classification governs Privacy review)",
            "reference": "/vendor-risk",
        })

    # 5. Department & Budget check
    dept = get_requester_department(request, fixtures_dir=fixtures_dir)
    budget_result = get_department_budget(dept or "", fixtures_dir=fixtures_dir)
    cost = request.get("annual_cost_usd")
    b_check = check_budget(cost, budget_result)

    missing_dept = b_check.get("missing_department", False)
    budget_insufficient = b_check.get("budget_insufficient", False)

    if missing_dept:
        if "missing_information" not in risk_flags:
            risk_flags.append("missing_information")
        if "Department budget record missing" not in missing_information:
            missing_information.append("Department budget record missing")
        evidence.append({
            "source": "department_budgets",
            "finding": f"Department budget record missing for '{dept or request.get('department')}'",
            "reference": "department_budgets.csv",
        })
    elif budget_insufficient:
        if "budget_insufficient" not in risk_flags:
            risk_flags.append("budget_insufficient")
        evidence.append({
            "source": "department_budgets",
            "finding": f"Annual cost ${cost} exceeds available budget ${budget_result.get('available_usd')}",
            "reference": "department_budgets.csv",
        })
    else:
        if cost is not None and budget_result.get("found"):
            evidence.append({
                "source": "department_budgets",
                "finding": f"Annual cost ${cost} is within available budget ${budget_result.get('available_usd')}",
                "reference": "department_budgets.csv",
            })

    # 6. Governance triggers & approval routing
    required_approvals: list[str] = []

    # When material fields are missing, review is paused: required_approvals is empty
    if missing_fields:
        required_approvals = []
    else:
        base_approvals = approval_thresholds(cost)
        required_approvals = list(base_approvals)

        # Judgment call: when department budget record is missing, add Finance approval
        if missing_dept and "Finance" not in required_approvals:
            required_approvals.append("Finance")

        data_access = clean_str(request.get("data_access_level"))
        integrations = [clean_str(i) for i in (request.get("requested_integrations") or [])]
        has_prod_cloud = any("production" in i or "cloud" in i for i in integrations)

        reg_sec_status = clean_str(registry_row.get("security_status") if registry_row else None) or clean_str(registry_row.get("security_review_status") if registry_row else None)
        api_sec_status = clean_str(vendor_api_result.get("security_review_status") if vendor_api_result else None)

        # Security review triggers per Policy Section 5
        sec_review = False
        if is_sensitive_data_access(data_access):
            sec_review = True
        elif has_prod_cloud:
            sec_review = True
        elif reconcile_res.get("is_expired"):
            sec_review = True
        elif registry_row is None and vendor_api_result is None:
            sec_review = True
        elif registry_row and reg_sec_status not in ("approved", "approved - limited use"):
            sec_review = True
        elif vendor_api_result and api_sec_status not in ("approved", "approved - limited use"):
            sec_review = True
        elif "vendor_risk_unavailable" in risk_flags:
            sec_review = True

        if sec_review:
            if "security_review_required" not in risk_flags:
                risk_flags.append("security_review_required")
            if "Security" not in required_approvals:
                required_approvals.append("Security")

        # Privacy review triggers per Policy Section 6
        # Triggers for employee/customer PII, or sensitive data stored outside region
        api_outside = clean_bool((vendor_api_result or {}).get("stores_data_outside_region"))
        reg_outside = clean_bool(registry_row.get("stores_data_outside_region") if registry_row else None)
        stores_outside = api_outside or reg_outside

        privacy_review = False
        if data_access in PII_DATA_ACCESS_LEVELS:
            privacy_review = True
        elif stores_outside and is_sensitive_data_access(data_access):
            privacy_review = True

        if privacy_review:
            if "privacy_review_required" not in risk_flags:
                risk_flags.append("privacy_review_required")
            if "Privacy" not in required_approvals:
                required_approvals.append("Privacy")

        # Legal review triggers per Policy Section 7
        # Precedence:
        # (1) Non-approved / non-standard legal terms (Draft, Unknown, Pending) trigger Legal review regardless of spend.
        # (2) New vendor with annual spend >= $10,000.
        # (3) Judgment call: new vendor with unavailable vendor risk API routes to Legal for contract diligence.
        # (4) Material cross-region transfer of sensitive/PII data on significant spend (>= $10,000).
        cost_val = Decimal(str(cost)) if cost is not None else Decimal("0")
        is_new_vendor = registry_row is None or clean_str(registry_row.get("procurement_status")) == "new"
        legal_status = clean_str(registry_row.get("legal_terms_status") if registry_row else None)

        legal_review = False
        if registry_row is not None and legal_status not in ("approved", "standard"):
            legal_review = True
        elif is_new_vendor and cost_val >= Decimal("10000"):
            legal_review = True
        elif is_new_vendor and "vendor_risk_unavailable" in risk_flags:
            legal_review = True
        elif stores_outside and data_access in PII_DATA_ACCESS_LEVELS and cost_val >= Decimal("10000"):
            legal_review = True

        if legal_review:
            if "legal_review_required" not in risk_flags:
                risk_flags.append("legal_review_required")
            if "Legal" not in required_approvals:
                required_approvals.append("Legal")

    # Determine if governance review is required
    # Note: Governance roles per Rule 9: Security, Privacy, Legal, Finance or CFO
    governance_roles = {"Security", "Privacy", "Legal", "Finance", "CFO"}
    governance_required = any(r in governance_roles for r in required_approvals)

    floor_category = compute_floor_category(
        missing_fields=missing_fields,
        has_unavailable_evidence="vendor_risk_unavailable" in risk_flags,
        budget_insufficient=budget_insufficient,
        missing_department=missing_dept,
        governance_required=governance_required,
    )

    return {
        "required_approvals": required_approvals,
        "risk_flags": risk_flags,
        "missing_information": missing_information,
        "recommendation_category_floor": floor_category,
        "evidence": evidence,
    }
