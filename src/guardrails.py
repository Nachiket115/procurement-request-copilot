from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any

from src.contracts import (
    EvidenceItem,
    ProcurementDecision,
    RecommendationCategory,
    RunTelemetry,
)
from src.rules import detect_injection, evaluate_request

CANONICAL_APPROVALS = [
    "Manager",
    "Department Head",
    "Procurement",
    "Finance",
    "CFO",
    "Security",
    "Privacy",
    "Legal",
]

CANONICAL_RISK_FLAGS = {
    "existing_tool_overlap",
    "budget_insufficient",
    "security_review_required",
    "privacy_review_required",
    "legal_review_required",
    "vendor_review_expired",
    "conflicting_vendor_evidence",
    "vendor_risk_unavailable",
    "prompt_injection_detected",
    "missing_information",
}

DEFAULT_RECOMMENDATION = {
    RecommendationCategory.route_for_standard_review: "Route request for standard manager and procurement review.",
    RecommendationCategory.route_for_governance_review: "Route request for required governance reviews (Security/Privacy/Legal/Finance/CFO).",
    RecommendationCategory.use_existing_tool: "Use existing internal software catalog tool instead of new purchase.",
    RecommendationCategory.escalate_to_finance: "Escalate request to Finance due to budget deficit or missing department allocation.",
    RecommendationCategory.escalate_unavailable_evidence: "Escalate request due to unavailable external vendor risk evidence.",
    RecommendationCategory.needs_more_information: "Request additional material information before proceeding with review.",
}

DEFAULT_NEXT_STEP = {
    RecommendationCategory.route_for_standard_review: "Submit request to designated approver(s).",
    RecommendationCategory.route_for_governance_review: "Coordinate with governance stakeholders for required sign-offs.",
    RecommendationCategory.use_existing_tool: "Notify requester of existing tool in software catalog.",
    RecommendationCategory.escalate_to_finance: "Contact Finance team for budget review or reallocation.",
    RecommendationCategory.escalate_unavailable_evidence: "Perform manual vendor risk diligence.",
    RecommendationCategory.needs_more_information: "Prompt requester for missing required procurement fields.",
}

# Patterns indicating the draft claims final or unilateral approval rather than advisory recommendation.
# Policy Section 11: The copilot is advisory-only and cannot grant final purchase approval.
CLAIMED_APPROVAL_PATTERNS = [
    re.compile(r"\bauto-approved?\b", re.IGNORECASE),
    re.compile(r"\bpurchase approved\b", re.IGNORECASE),
    re.compile(r"\brequest (?:is )?approved\b", re.IGNORECASE),
    re.compile(r"\bapproval granted\b", re.IGNORECASE),
    re.compile(r"\bapprove immediately\b", re.IGNORECASE),
    re.compile(r"\bpre-approved\b", re.IGNORECASE),
    re.compile(r"\b(request|purchase|tool|license|software|order)\s+(is|has been)\s+(now\s+)?approved\b", re.IGNORECASE),
    re.compile(r"^approved\.?$", re.IGNORECASE),
    re.compile(r"^auto-approved\.?$", re.IGNORECASE),
]


def detect_claimed_approval(text: str | None) -> bool:
    """Detect whether LLM draft text claims final or unilateral approval instead of advisory recommendation."""
    if not text or not str(text).strip():
        return False
    t = str(text).strip()
    low = t.lower()

    for p in CLAIMED_APPROVAL_PATTERNS:
        if p.search(t):
            return True

    if low.rstrip(".!") in ("approved", "approve", "auto-approved", "auto-approve"):
        return True

    return False


def apply_guardrails(
    draft: dict | None,
    request: dict,
    vendor_api_result: dict | None = None,
    started_at: float | None = None,
    llm_calls: int = 0,
    tool_calls: int = 0,
    tool_names: list[str] | None = None,
    model_name: str | None = None,
    fixtures_dir: Path | str | None = None,
) -> ProcurementDecision:
    """Apply deterministic safety and policy guardrails to an LLM draft response.

    When fixtures_dir is None, the existing FIXTURES_DIR env fallback in data_access still applies.
    """
    now = time.time()
    if started_at is None:
        started_at = now

    req_id = str(request.get("request_id", "UNKNOWN-REQ"))

    # 1. Deterministic floor evaluation
    floor = evaluate_request(
        request,
        vendor_api_result=vendor_api_result,
        fixtures_dir=fixtures_dir,
    )
    floor_approvals = list(floor.get("required_approvals", []))
    orig_floor_flags = list(floor.get("risk_flags", []))
    floor_flags = list(orig_floor_flags)
    floor_missing = list(floor.get("missing_information", []))
    floor_cat_str = str(floor.get("recommendation_category_floor", RecommendationCategory.route_for_standard_review.value))
    floor_evidence = list(floor.get("evidence", []))

    llm_fallback = False
    draft_rejected_injection = False
    draft_claimed_approval = False

    if not isinstance(draft, dict):
        draft_dict: dict[str, Any] = {}
        llm_fallback = True
    else:
        draft_dict = draft

    # Check for prompt injection in draft text
    draft_rec_raw = draft_dict.get("recommendation", "")
    draft_next_raw = draft_dict.get("next_step", "")
    is_inj_rec, _ = detect_injection(str(draft_rec_raw)) if draft_rec_raw else (False, None)
    is_inj_next, _ = detect_injection(str(draft_next_raw)) if draft_next_raw else (False, None)
    if is_inj_rec or is_inj_next:
        if "prompt_injection_detected" not in floor_flags:
            floor_flags.append("prompt_injection_detected")
        draft_rec_raw = None
        draft_next_raw = None
        draft_rejected_injection = True
    elif detect_claimed_approval(draft_rec_raw) or detect_claimed_approval(draft_next_raw):
        draft_rec_raw = None
        draft_next_raw = None
        draft_claimed_approval = True

    # 2. Risk flags: floor flags UNION draft flags restricted to CANONICAL_RISK_FLAGS (floor order first, then extra draft flags)
    # Over-escalation is the safe failure for a compliance tool, so valid extra approvals/flags from the LLM are kept.
    draft_flags_raw = draft_dict.get("risk_flags", []) if isinstance(draft_dict.get("risk_flags"), list) else []
    canonical_draft_flags = [f for f in draft_flags_raw if isinstance(f, str) and f in CANONICAL_RISK_FLAGS]

    final_flags = list(floor_flags)
    for f in canonical_draft_flags:
        if f not in final_flags:
            final_flags.append(f)

    # 3. Missing information: floor's strictly
    final_missing = list(floor_missing)

    # 4. Approvals: floor approvals UNION draft approvals restricted to canonical set, output in canonical order.
    # If the floor has missing material fields, approvals = [] regardless of the draft.
    # Over-escalation is the safe failure for a compliance tool, so valid extra approvals from the LLM are kept.
    if final_missing or floor_cat_str == RecommendationCategory.needs_more_information.value:
        final_approvals: list[str] = []
    else:
        draft_appr_raw = draft_dict.get("required_approvals", []) if isinstance(draft_dict.get("required_approvals"), list) else []
        canonical_draft_appr = [a for a in draft_appr_raw if isinstance(a, str) and a in CANONICAL_APPROVALS]
        combined_appr_set = set(floor_approvals) | set(canonical_draft_appr)
        final_approvals = [a for a in CANONICAL_APPROVALS if a in combined_appr_set]

    # 5. Recommendation category: floor category with standard -> use_existing_tool upgrade only if overlap present in floor
    draft_cat_raw = draft_dict.get("recommendation_category")
    if isinstance(draft_cat_raw, RecommendationCategory):
        draft_cat_val = draft_cat_raw.value
    elif isinstance(draft_cat_raw, str):
        draft_cat_val = draft_cat_raw
    else:
        draft_cat_val = None

    if (
        floor_cat_str == RecommendationCategory.route_for_standard_review.value
        and draft_cat_val == RecommendationCategory.use_existing_tool.value
        and "existing_tool_overlap" in floor_flags
    ):
        final_category = RecommendationCategory.use_existing_tool
    else:
        try:
            final_category = RecommendationCategory(floor_cat_str)
        except ValueError:
            final_category = RecommendationCategory.route_for_standard_review

    # 6. Human review required (always True by policy)
    final_human_review = True

    # 7. Evidence: floor evidence + valid draft evidence
    evidence_items: list[EvidenceItem] = []
    seen_evidence = set()

    for e in floor_evidence:
        if isinstance(e, dict):
            src = str(e.get("source", "rules_engine"))
            fnd = str(e.get("finding", ""))
            ref = str(e.get("reference")) if e.get("reference") is not None else None
        elif isinstance(e, EvidenceItem):
            src = e.source
            fnd = e.finding
            ref = e.reference
        else:
            continue
        key = (src, fnd)
        if key not in seen_evidence:
            seen_evidence.add(key)
            evidence_items.append(EvidenceItem(source=src, finding=fnd, reference=ref))

    if isinstance(draft_dict.get("evidence"), list):
        for e in draft_dict["evidence"]:
            if isinstance(e, dict) and "source" in e and "finding" in e:
                fnd = str(e["finding"])
                is_inj, _ = detect_injection(fnd)
                if is_inj:
                    fnd = "[WITHHELD: prompt injection pattern detected]"
                src = str(e["source"])
                ref = str(e.get("reference")) if e.get("reference") is not None else None
                key = (src, fnd)
                if key not in seen_evidence:
                    seen_evidence.add(key)
                    evidence_items.append(EvidenceItem(source=src, finding=fnd, reference=ref))
            elif isinstance(e, EvidenceItem):
                key = (e.source, e.finding)
                if key not in seen_evidence:
                    seen_evidence.add(key)
                    evidence_items.append(e)

    # 8. Recommendation text and next_step
    if not llm_fallback and not draft_rejected_injection and not draft_claimed_approval and draft_rec_raw and isinstance(draft_rec_raw, str) and draft_rec_raw.strip():
        recommendation_text = draft_rec_raw.strip()
    else:
        recommendation_text = DEFAULT_RECOMMENDATION.get(
            final_category,
            "Evaluate request per procurement policy.",
        )

    if not llm_fallback and not draft_rejected_injection and not draft_claimed_approval and draft_next_raw and isinstance(draft_next_raw, str) and draft_next_raw.strip():
        next_step_text = draft_next_raw.strip()
    else:
        default_next = DEFAULT_NEXT_STEP.get(
            final_category,
            "Submit request for standard processing.",
        )
        if llm_fallback:
            next_step_text = f"{default_next} [llm_fallback_engaged]"
        elif draft_rejected_injection:
            next_step_text = f"{default_next} [draft_rejected_injection]"
        elif draft_claimed_approval:
            next_step_text = f"{default_next} [draft_claimed_approval]"
        else:
            next_step_text = default_next

    # 9. Guardrail correction counts:
    # removals: invalid approval/flag names the draft listed, category overridden, human_review_required False, draft missing_information not in the floor.
    # A valid canonical approval/flag the draft added beyond the floor is KEPT and counts as neither a removal nor an addition.
    # additions: floor approvals/flags/missing items the draft omitted (using original floor flags before guardrail scanner modifications).
    # When draft is None or not a dict (llm_fallback), additions = 0 and removals = 0.
    if llm_fallback:
        additions = 0
        removals = 0
    else:
        draft_appr = draft_dict.get("required_approvals", []) if isinstance(draft_dict.get("required_approvals"), list) else []
        draft_flags = draft_dict.get("risk_flags", []) if isinstance(draft_dict.get("risk_flags"), list) else []
        draft_missing = draft_dict.get("missing_information", []) if isinstance(draft_dict.get("missing_information"), list) else []

        removals = 0
        # Invalid approval names (or all approvals if missing fields cleared approvals)
        if final_missing or floor_cat_str == RecommendationCategory.needs_more_information.value:
            removals += len(draft_appr)
        else:
            removals += len([a for a in draft_appr if a not in CANONICAL_APPROVALS])

        # Invalid risk flag names
        removals += len([f for f in draft_flags if f not in CANONICAL_RISK_FLAGS])

        # Draft missing_information not in floor
        removals += len([m for m in draft_missing if m not in final_missing])

        # Category overridden
        if draft_cat_val is not None and draft_cat_val != final_category.value:
            removals += 1

        # Human review required overridden from False to True
        if draft_dict.get("human_review_required") is False:
            removals += 1

        additions = 0
        # Floor approvals omitted by draft (only if floor actually has approvals)
        if not final_missing and floor_cat_str != RecommendationCategory.needs_more_information.value:
            additions += len([a for a in floor_approvals if a not in draft_appr])

        # Floor flags omitted by draft (computed against original floor flags)
        additions += len([f for f in orig_floor_flags if f not in draft_flags])

        # Floor missing info omitted by draft
        additions += len([m for m in floor_missing if m not in draft_missing])

    corrections = additions + removals

    # 10. Telemetry
    latency_ms = (time.time() - started_at) * 1000.0
    telemetry = RunTelemetry(
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        tool_names=list(tool_names) if tool_names else [],
        latency_ms=round(latency_ms, 2),
        guardrail_corrections=corrections,
        guardrail_additions=additions,
        guardrail_removals=removals,
        model_name=model_name,
    )

    return ProcurementDecision(
        request_id=req_id,
        recommendation=recommendation_text,
        recommendation_category=final_category,
        evidence=evidence_items,
        required_approvals=final_approvals,
        missing_information=final_missing,
        risk_flags=final_flags,
        next_step=next_step_text,
        human_review_required=final_human_review,
        telemetry=telemetry,
    )
