# Architectural & Policy Assumptions

This document outlines the operational and policy assumptions underpinning the deterministic procurement policy engine and copilot workflows.

---

## 1. Reference Date & Time Horizon
- **Authoritative Date:** The evaluation reference date is fixed at `2026-09-30`. Calls to dynamic system dates (e.g. `datetime.date.today()`) are prohibited to guarantee reproducible assessments across historical audit runs.
- **Vendor Assessment Expiry:** Assessments are valid for exactly 365 days (`(REFERENCE_DATE - review_date).days <= 365`). An assessment dated `2025-09-30` is valid on `2026-09-30`; an assessment dated `2025-09-29` (366 days) is expired.

---

## 2. Catalog Overlap Rules
- **Literal Policy Section 3 Overlap:** An `existing_tool_overlap` risk flag is raised if the request matches an existing catalog entry by:
  - Exact or partial product name,
  - Same vendor name, or
  - Same product category.
- **Informational Overlap Type:** The classified `overlap_type` (`duplicate`, `expansion`, `competitor`) is surfaced for reviewer context and evidence logging only; it does not deterministically override approval tiers.

---

## 3. Governance Review Triggers

### 3.1 Security Review
- Triggered if `data_access_level` involves sensitive data classes (`source_code`, `confidential_documents`, `employee_pii`, `customer_pii`, `credentials/secrets`, `production_telemetry`, or keywords `production`, `cloud`, `credential`, `secret`).
- Triggered if integrations involve production or cloud environments.
- Triggered if internal registry or vendor risk API security status is not approved (e.g., Pending, Unknown, Expired, missing), or if the risk API is unavailable.

### 3.2 Privacy Review
- Triggered when request data access involves personal data (`employee_pii` or `customer_pii`), or sensitive data transferred to a vendor with `stores_data_outside_region: True`.
- **Vendor-Level Personal Data:** Vendor `processes_personal_data: True` is captured as an evidence note only; it does not trigger a Privacy review unless the specific request's data class involves PII or out-of-region sensitive data. For example, REQ-1002 (BrandBoard Enterprise) accesses `internal_marketing` data and deliberately does **not** trigger a Privacy review, even though the vendor risk API notes that the vendor processes personal data.

### 3.3 Legal Review Precedence
Legal review routing follows strict precedence:
1. **Non-Standard Terms:** Legal terms not marked `Approved` or `Standard` (e.g., `Draft`, `Unknown`, `Pending`, `NaN`) trigger Legal review regardless of annual spend.
2. **New Vendor Spend Threshold:** Any new vendor with annual spend $\ge \$10,000$ requires Legal review.
3. **Cross-Region PII Transfer:** Significant spend ($\ge \$10,000$) involving cross-region PII data transfers requires Legal review.
4. **Judgment Call (New Vendor with Unavailable Risk API):** A new vendor whose risk/assessment API is unreachable (`vendor_risk_unavailable`) is routed to Legal for contract terms verification.

---

## 4. Financial Routing & Department Budget Allocation
- **Missing Department Budget (Judgment Call):** If a requesting department is not present in `department_budgets.csv` (e.g., "Go To Market"), the copilot adds `Finance` to `required_approvals` to review budget allocation rather than raising an unhandled exception or prematurely marking `budget_insufficient`.
- **Unresolved Material Fields:** If required fields (`vendor_name`, `annual_cost_usd`, `user_count`, or `data_access_level`) are missing, review is paused: `required_approvals` is set to `[]`, and `needs_more_information` takes precedence over all governance and financial triggers.

---

## 5. Security Scanning & Untrusted Input Handling
- **Prompt Injection:** Heuristic injection pattern detection flags `prompt_injection_detected` in `risk_flags` without altering the deterministic recommendation category floor or approval thresholds.
- **Recommendation Upgrades (`use_existing_tool`):** `use_existing_tool` is an LLM-only upgrade from the `route_for_standard_review` baseline floor. Requests requiring governance (Security, Privacy, Legal, Finance) remain at `route_for_governance_review` or escalation categories.
- **Unknown Vendor (404):** A 404 response from the vendor risk service is treated as unavailable evidence (`vendor_risk_unavailable`).

---

## 6. Contracts & Evaluation Schema
- **Recommendation Categories:** Modeled via the `RecommendationCategory` enum (`route_for_standard_review`, `route_for_governance_review`, `use_existing_tool`, `escalate_to_finance`, `escalate_unavailable_evidence`, `needs_more_information`).
- **Approval Comparison:** The evaluation runner will compare approvals as sets (order-agnostic) while preserving required governance roles.
