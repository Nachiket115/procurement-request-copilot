# Workflow & Architecture Notes

This document describes the end-to-end procurement copilot workflow, tool responsibilities, agent architecture, handoff formats, and intentional scope boundaries.

---

## Workflow Diagram

```text
Employee Request
      |
      v
 Understand Request ──> Resolve requester identity & department (data_access.py)
      |                  Scan justification for prompt injection (rules.py)
      |
      v
 Gather Evidence ─────> check_department_budget()     [CODE, tamper-resistant]
      |                  check_software_catalog()      [CODE, model-supplied args]
      |                  verify_vendor_risk()           [CODE, model-supplied args]
      |                  compute_required_approvals()   [CODE, tamper-resistant]
      |
      v
 Recommend ───────────> LLM synthesizes structured draft (single_agent / staged)
      |                  Post-LLM guardrail unions draft with deterministic floor
      |
      v
 Human Review ────────> Reviewer sees advisory ProcurementDecision
                         Approve / Request more info / Escalate (in-memory only)
```

---

## Deterministic (CODE) vs Model-Driven (LLM) vs Human

| Responsibility | Owner | Examples |
|---|---|---|
| Financial thresholds, approval tiers, compliance floors | **CODE** | Dollar tier routing, `budget_insufficient`, mandatory approvers |
| Catalog overlap detection, vendor registry reconciliation | **CODE** | `existing_tool_overlap`, `vendor_review_expired` flags |
| Injection scanning, evidence sanitation | **CODE** | Regex heuristics, XML delimiters, notes withheld |
| Business context interpretation, ambiguity resolution | **LLM** | Natural-language rationale, `use_existing_tool` judgment |
| Final sign-off, escalation handling | **HUMAN** | Approve / reject / escalate via dashboard buttons |

---

## Architecture A vs B

- **Architecture A (Single Agent, Shipped):** One LLM loop calls all 4 bound tools, produces a structured draft, which the post-LLM guardrail engine validates against the deterministic floor.
- **Architecture B (Staged / 2-Agent):** Stage 1 Analyst runs tools and builds raw evidence. A sanitization layer strips injection patterns and raw justification. Stage 2 Reviewer (no tool access, blind to raw justification) generates the draft. Both pass through the same post-LLM guardrail.

---

## Handoff Formats

- **Inter-stage (Architecture B):** Sanitized `EvidencePack` — structured tool outputs with adversarial text stripped and the deterministic floor pre-computed.
- **Final output (both architectures):** `ProcurementDecision` Pydantic schema containing `request_id`, `recommendation`, `recommendation_category`, `evidence`, `required_approvals`, `missing_information`, `risk_flags`, `next_step`, `human_review_required` (always `True`), and `telemetry`.

---

## Stop & Escalation Conditions

1. **Missing material fields** (`vendor_name`, `annual_cost_usd`, `user_count`, `data_access_level`): approval paused, `required_approvals = []`, category set to `needs_more_information`.
2. **Vendor risk API unavailable** (503/404/timeout): flags `vendor_risk_unavailable`, escalates to Security & Legal.
3. **Prompt injection detected:** flags `prompt_injection_detected`, sanitizes pre-approved claims; does not alter financial tier.
4. **LLM failure/timeout:** deterministic fallback engages, tags `[llm_fallback_engaged]` in `next_step`.

---

## What We Intentionally Did Not Build

- No real purchasing system integration (reviewer actions are in-memory session state only).
- No semantic injection classifier (regex heuristics only; production requires ML-based detection).
- No live SaaS seat utilization telemetry (static catalog CSV only).
- No multi-model or multi-run variance estimation (single model, n=1 per case).
- No production API quota management (mock vendor-risk service, free-tier Gemini).
