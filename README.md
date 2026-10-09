# AI Procurement Request Copilot

An enterprise procurement copilot that inspects software and service purchase requests, gathers evidence from internal databases and external vendor-risk APIs, enforces deterministic financial and compliance policy floors, detects prompt injection attempts, and delivers structured advisory recommendations for human review.

- **Architecture Decision:** [Architecture Decision Memo](templates/architecture_decision.md) (recommends shipping Architecture A)
- **Supporting Documentation:** [Policy & Architectural Assumptions](docs/ASSUMPTIONS.md) | [Bugs Fixed in Starter Pack](docs/BUGS_FIXED.md) | [Known Technical Limitations](docs/KNOWN_LIMITATIONS.md) | [Starter Pack README](docs/STARTER_README.md)

---

## 1. Quick Start

### 1.1 Clone and Environment Setup

```bash
git clone https://github.com/Nachiket115/procurement-request-copilot.git
cd procurement-request-copilot

python3.12 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 1.2 Configuration

Create `.env` from `.env.example` and provide your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
VENDOR_RISK_BASE_URL=http://127.0.0.1:8001
GOOGLE_API_KEY=your-gemini-api-key-here
MODEL_NAME=gemini-3.1-flash-lite
```
*(Get a free API key at [aistudio.google.com](https://aistudio.google.com). Note: `gemini-2.5-flash-lite` has a strict 20-requests/day limit on the free tier discovered from the AI Studio rate limits documentation; `gemini-3.1-flash-lite` provides 15 RPM and 500 RPD, which is why `MODEL_NAME` is configurable.)*

### 1.3 Run Local Services & Web UI

Start both the mock vendor-risk API (port `8001` with synthetic fixture vendors loaded via `VENDOR_RISK_FIXTURES_PATH`) and the Streamlit 3-panel UI (port `8501`) with a single command:

```bash
python run_local.py
```

Open your browser to: **`http://localhost:8501`**

### 1.4 Run Automated Tests

Execute the comprehensive offline unit and integration test suite (92 tests, runs in <0.5s with zero LLM or network calls):

```bash
python -m unittest discover -s tests -v
```

### 1.5 Run Evaluations

Run the end-to-end benchmark across all 21 test cases (10 starter + 11 synthetic edge cases):

```bash
# Instant deterministic policy engine evaluation (no API key required, runs in ~2 ms):
python evals/run_evals.py --architecture deterministic

# Full evaluation across all architectures (~20 min on Gemini free tier, ~150 LLM calls):
python evals/run_evals.py --architecture all
```

*Note on Fixtures & Port Isolation:* Synthetic edge cases reside in `evals/fixtures/` and are loaded only when opted in via `fixtures_dir` or the `FIXTURES_DIR` environment variable. When executing evaluations, `evals/run_evals.py` launches its own dedicated mock API instance on port `8002` with fixtures merged, keeping the evaluation environment completely isolated from any local development servers on port `8001`.

---

## 2. Product Workflow & UI

The copilot implements an advisory workflow where software purchase requests are analyzed and synthesized before being placed before a human approver:

```text
Employee Request ──> Understand ──> Gather Evidence ──> Recommend ──> Human Review
```

### 3-Panel Streamlit Dashboard

1. **Panel 1: Request Details:** Displays requester identity and department (resolved via `src/data_access.py`), product, vendor, estimated annual spend, seat count, data access classification, requested integrations, urgency, and the raw business justification rendered in a quote block. Justifications are actively scanned for prompt injection; if triggered, a red alert badge (`🚨 possible prompt injection`) is displayed.
2. **Panel 2: Evidence & Risks:** Shows color-coded risk flags (Security, Privacy, Legal, Injection, Budget, Overlap, Expiration, Vendor Unavailable), missing required information items, and collated evidence items grouped by source (Software Catalog, Department Budgets, Vendor Registry, Vendor Risk API) with verifiable citations and policy references.
3. **Panel 3: Recommendation & Action:** Displays the recommendation category badge, structured rationale, an ordered checklist of required human approvers with checkbox glyphs (`☐`), and recommended next operational steps.
   - **UI Note on Deterministic Runs:** On deterministic runs, the yellow "LLM unavailable: deterministic fallback shown" banner is hidden by design (the deterministic engine appends `[llm_fallback_engaged]` internally to indicate rules-only execution without model involvement).
   - **Reviewer Action Buttons:** Reviewers can click **Approve**, **Request more info**, or **Escalate** with an optional comment. These buttons record the human reviewer's choice only into an in-memory session decision log; nothing is sent to any external purchasing system.
   - **Telemetry:** Displays execution architecture (from the cached run), model name, latency, LLM turns, tool calls, and post-LLM guardrail adjustments.

### Interface Screenshots

![REQ-1001: Low-value standard purchase request](docs/screenshots/req-1001.png)
*Figure 1: REQ-1001 — Low-value e-signature add-on ($800/yr), auto-routed for standard Manager approval.*

![REQ-1006: Prompt injection attack with missing information](docs/screenshots/req-1006.png)
*Figure 2: REQ-1006 — Adversarial prompt injection detected in justification text; material fields missing, paused for review.*

---

## 3. Architecture & Design Decisions

### Core Design Principle

| Layer | Responsibility | Operating Principle |
|---|---|---|
| **AI (LLM)** | Interprets business context, reasons over qualitative nuances, evaluates existing tool alternatives | Recommends, never purchases |
| **CODE (Python)** | Enforces financial thresholds, validates catalog overlaps, computes deterministic compliance floors | Enforces, never guesses |
| **HUMAN** | Validates advisory output, provides final sign-off, handles escalations | Decides, always in the loop |

### Architecture A: Single-Agent Baseline (Shipped)

A tool-calling loop where a single agent inspects the request, calls bound deterministic tools, and produces a structured draft validated by post-LLM guardrails.

```mermaid
flowchart TD
    Req[Purchase Request] --> Prefetch[Vendor Prefetch API]
    Prefetch --> SingleAgent[Single Agent LLM Loop]
    subgraph BoundTools[Bound Deterministic Tools]
        T1[check_department_budget]
        T2[check_software_catalog]
        T3[verify_vendor_risk]
        T4[compute_required_approvals]
    end
    SingleAgent <--> BoundTools
    SingleAgent --> Draft[Structured Draft Output]
    Draft --> Guardrail[Post-LLM Guardrail Engine]
    Floor[Deterministic Floor Engine] --> Guardrail
    Guardrail --> Decision[ProcurementDecision<br/>human_review_required=True]
```

### Architecture B: Staged / 2-Agent Variant

Separates evidence gathering from recommendation reasoning. The Analyst agent runs tool loops and builds an evidence pack; an isolation layer sanitizes adversarial text; the Reviewer agent (with zero tool access) generates the draft recommendation.

```mermaid
flowchart TD
    Req[Purchase Request] --> Prefetch[Vendor Prefetch API]
    Prefetch --> Analyst[Stage 1: Analyst Agent]
    subgraph AnalystTools[Analyst Bound Tools]
        AT1[check_department_budget]
        AT2[check_software_catalog]
        AT3[verify_vendor_risk]
        AT4[compute_required_approvals]
    end
    Analyst <--> AnalystTools
    Analyst --> RawPack[Raw Tool Evidence]
    RawPack --> Sanitize[Sanitization Layer<br/>Strips Injection Patterns & Raw Justification]
    Sanitize --> EvidencePack[Sanitized EvidencePack]
    EvidencePack --> Reviewer[Stage 2: Reviewer Agent<br/>No Tools, Blind to Raw Justification]
    Floor[Deterministic Floor Engine] --> Reviewer
    Reviewer --> Draft[Structured Draft Output]
    Draft --> Guardrail[Post-LLM Guardrail Engine]
    Floor --> Guardrail
    Guardrail --> Decision[ProcurementDecision<br/>human_review_required=True]
```

### Key Architectural & Design Decisions

1. **Separation of Tool Calling and Structured Extraction:** Tool use turns and structured output formatting are executed as distinct Gemini API calls. The model first reasons and interacts with tools in standard conversation turns, followed by a constrained structured extraction call against the Pydantic schema.
2. **Deterministic Vendor Prefetching for Parity:** The evaluation harness (not the model) prefetches external vendor data once per request before agent invocation and counts it as 1 tool call across all architectures. This guarantees strict telemetry parity.
3. **Tamper-Resistant Bound Tools:** Tools `compute_required_approvals` and `check_department_budget` are bound to the verified request payload in closure. The LLM cannot alter budget amounts or fake department names during tool execution.
4. **Staged Reviewer Isolation:** In Architecture B, the Stage 2 Reviewer agent never sees the raw untrusted business justification text, preventing direct prompt injection at the final decision stage.
5. **Offline Mock Testing:** The test suite (92 tests) relies entirely on deterministic unit tests and mocks; no real LLM calls are made during tests. Real LLM behavior was validated via smoke tests and the evaluation benchmark.

### File Map

```text
├── app.py                   # Streamlit 3-panel interactive copilot dashboard
├── run_local.py             # One-command launcher for mock API (8001) + Streamlit UI (8501)
├── src/
│   ├── contracts.py         # Pydantic schemas (ProcurementDecision, RunTelemetry, EvidenceItem)
│   ├── rules.py             # Deterministic policy rules, threshold tables, injection heuristics, floor engine
│   ├── guardrails.py        # Post-LLM guardrails, floor union, claim sanitation, fallback tagging
│   ├── tools.py             # 4 tamper-resistant bound tools exposed to agent loops
│   ├── llm.py               # Google GenAI Gemini SDK client, retry backoff, disk caching, structured output
│   ├── single_agent.py      # Architecture A: Single-agent tool loop and decision synthesizer
│   ├── staged_agents.py     # Architecture B: Staged Analyst -> Sanitizer -> Reviewer pipeline
│   ├── solution.py          # Unified entrypoint adapter handle_request(request_or_id, architecture)
│   ├── data_access.py       # Cached loaders for employees, department budgets, catalogs, and vendors
│   ├── vendor_client.py     # Resilient HTTP client for the external mock vendor-risk API
│   └── ui_helpers.py        # Pure UI helper functions (color coding, badge labels, evidence grouping)
├── mock_api/
│   └── app.py               # FastAPI mock vendor risk service
├── evals/
│   ├── ground_truth.json    # Authoritative expectations for 21 test cases
│   ├── run_evals.py         # Automated evaluation benchmark runner
│   ├── fixtures/            # 11 synthetic edge cases and fixture vendor database
│   └── results/             # Evaluation summary and per-case CSV outputs
├── docs/
│   ├── ASSUMPTIONS.md       # Operational assumptions and policy decisions
│   ├── BUGS_FIXED.md        # Defects identified and resolved in the starter code
│   ├── KNOWN_LIMITATIONS.md # Technical boundaries and heuristic trade-offs
│   ├── STARTER_README.md    # Original starter pack README
│   └── screenshots/         # UI dashboard verification screenshots
└── templates/
    └── architecture_decision.md # Architecture decision memo
```

---

## 4. Tools, Policies, and Contracts

### The 4 Bound Tools

1. `check_department_budget()`: Verifies available annual software budget. Takes no model-supplied arguments (bound to the real request via `bind_tools` so the model cannot alter budget inputs).
2. `check_software_catalog(product_name, vendor_name, category)`: Scans catalog for overlapping or competitor solutions. Arguments are model-supplied.
3. `verify_vendor_risk(vendor_name)`: Queries external vendor risk API and reconciles the API result with the internal `vendors.csv` registry. Argument is model-supplied. The harness prefetches vendor data once per request (counted as 1 tool call, tool name `get_vendor_risk_prefetch`, in every architecture).
4. `compute_required_approvals()`: Deterministically calculates required management approvals per financial policy. Takes no model-supplied arguments (bound to the real request via `bind_tools`).

### Financial Approval Tiers (Policy Section 4)

| Annual Spend (USD) | Required Approvals |
|---|---|
| ≤ $1,000 | Manager |
| $1,000.01 – $10,000 | Department Head, Procurement |
| $10,000.01 – $25,000 | Department Head, Finance, Procurement |
| > $25,000 | Department Head, Finance, CFO, Procurement |

### Taxonomies & Schema

- **Canonical Approval Roles:** `Manager`, `Department Head`, `Procurement`, `Finance`, `CFO`, `Security`, `Privacy`, `Legal`.
- **Standard Risk Flags:** `existing_tool_overlap`, `budget_insufficient`, `security_review_required`, `privacy_review_required`, `legal_review_required`, `vendor_review_expired`, `conflicting_vendor_evidence`, `vendor_risk_unavailable`, `prompt_injection_detected`, `missing_information`.
- **Evaluation Reference Date:** Fixed at **`2026-09-30`**. Vendor assessments older than 365 days are flagged as `vendor_review_expired`.
- **Output Schema (`ProcurementDecision`):** `request_id`, `recommendation`, `recommendation_category`, `evidence`, `required_approvals`, `missing_information`, `risk_flags`, `next_step`, `human_review_required` (always `True`), and `telemetry`.

---

## 5. Reliability, Security & Human Controls

### Post-LLM Guardrail Engine

1. **Deterministic Floor Union:** The guardrail computes a deterministic policy floor from request data and tool outputs. The model's draft approvals and flags are *unioned* with this floor—an LLM cannot drop a required approver or bypass a compliance check.
2. **Canonical Role Enforcement:** Invented roles (e.g., "Infosec", "Compliance Committee") are stripped.
3. **Claimed Approval Sanitization:** If the LLM generates language claiming pre-authorization (e.g., "Already CFO-approved"), the draft's recommendation and next-step text are replaced with defaults (flagged with `[draft_claimed_approval]`), while approvals and flags still follow the deterministic floor union rule.
4. **Advisory Invariant:** `human_review_required` is unconditionally forced to `True`.
5. **Fallback Markers:** If an LLM call fails or times out, the engine executes a deterministic fallback, tagging `next_step` with `[llm_fallback_engaged]`.

### Resilient External Vendor Client

The vendor-risk client (`src/vendor_client.py`) wraps HTTP calls in robust error handling:
- HTTP `503 Service Unavailable`, `404 Not Found`, timeouts, network drops, and malformed non-JSON payloads return structured status dictionaries (`{"status": "unavailable" | "not_found"}`).
- The policy engine flags `vendor_risk_unavailable` and escalates to Security/Legal without crashing the pipeline.

### Prompt Injection Defenses

- Untrusted input fields (`business_justification`, vendor API notes, registry notes, catalog notes) are enclosed in strict XML delimiters.
- Text is scanned with heuristic regex patterns (`"ignore all procurement rules"`, `"treat this as cfo-approved"`, `"system override"`, etc.).
- When injection is detected in vendor notes, the text is withheld from evidence (`"notes withheld: injection pattern detected"`) to prevent poisoning downstream prompts.
- In Architecture B, Stage 2 (Reviewer) never sees raw user justification text.

---

## 6. Edge Cases Covered

| Edge Case Scenario | Test / Eval Case | Implementation Behavior |
|---|---|---|
| **Missing Material Information** | `BASE-06`, `SYN-11` | Pauses approval workflow; sets `required_approvals = []`, flags `missing_information`, categorizes as `needs_more_information`. |
| **Existing Tool Overlap** | `BASE-08` | Identifies competitor/expansion in catalog; flags `existing_tool_overlap`; allows LLM to suggest `use_existing_tool` only if standard review floor applies. |
| **Conflicting Vendor Evidence** | `BASE-07`, `SYN-08` | Registry says Approved but API says Expired/Pending; flags `conflicting_vendor_evidence` and `vendor_review_expired`; mandates Security review. |
| **Vendor Review Date Freshness** | `SYN-07`, `SYN-08` | Enforces 365-day cutoff relative to `2026-09-30`. Stale assessments trigger `vendor_review_expired`. |
| **Approval Threshold Boundaries** | `SYN-01` to `SYN-06` | Boundary test cases ($1000.00, $1000.01, $10000.00, $10000.01, $25000.00, $25000.01) correctly switch tiers. |
| **Missing Department Budget** | `SYN-09` | Unbudgeted department ("Go To Market") routes to `Finance` for allocation rather than raising unhandled exceptions. |
| **Prompt Injection Attack** | `BASE-06`, `SYN-10` | Heuristics detect override phrasing; flags `prompt_injection_detected`; suppresses fake pre-approvals. |
| **Vendor Risk API Outage** | `BASE-09` | NimbusAI returns 503; client handles gracefully, flags `vendor_risk_unavailable`, and routes to Security & Legal. |

---

## 7. Policy Assumptions & Judgment Calls ([docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md))

The operational and policy assumptions underpinning the copilot include the following explicit judgment calls:

1. **Literal Overlap Rule:** Following literal Policy Section 3, matching on product name, vendor name, or category means nearly every request matching the software catalog receives the `existing_tool_overlap` risk flag. The classified `overlap_type` (`duplicate`, `expansion`, `competitor`) is provided for informational context only.
2. **Request-Governed Privacy Triggers:** Privacy review is triggered strictly by the request's specific data access level (`employee_pii`, `customer_pii`, or sensitive data transferred cross-region to an out-of-region vendor), **not** by the vendor-level flag `processes_personal_data: True`. For example, REQ-1002 (BrandBoard Enterprise) accesses `internal_marketing` data and deliberately does **not** trigger a Privacy review, even though the vendor risk API notes that the vendor processes personal data.
3. **Legal Review on Non-Standard Terms:** Non-standard, draft, or unknown legal terms trigger Legal review regardless of annual spend.
4. **Missing Department Budget Allocation:** When a department lacks a budget line item (e.g., "Go To Market"), the copilot adds `Finance` to `required_approvals` to allocate funds rather than throwing an exception or falsely flagging `budget_insufficient`.
5. **Paused Review on Missing Material Fields:** If required fields (`vendor_name`, `annual_cost_usd`, `user_count`, `data_access_level`) are missing, the approval workflow is paused (`required_approvals = []`), and the request is categorized as `needs_more_information`.
6. **Prompt Injection Boundary:** Heuristic injection detection adds `prompt_injection_detected` to `risk_flags` and sanitizes pre-approved claims, but does **not** arbitrarily alter the financial approval tier or recommendation category floor.
7. **LLM-Only Upgrade on `use_existing_tool`:** `use_existing_tool` is an LLM-only upgrade from the `route_for_standard_review` baseline floor when an overlapping tool already satisfies business needs. Requests requiring governance (Security, Privacy, Legal, Finance) remain at `route_for_governance_review` or escalation categories. Consequently, **`BASE-08` (REQ-1008) is the one case where the deterministic engine and LLM recommendations differ** (`route_for_standard_review` vs `use_existing_tool`).
8. **Unknown Vendor Treated as Unavailable:** A `404 Not Found` response from the vendor risk API is treated as unavailable evidence (`vendor_risk_unavailable`), escalating to Security and Legal.
9. **Authoritative Snapshot Date:** Fixed at `2026-09-30` across all historical evaluations.

---

## 8. Starter Pack Defects Resolved ([docs/BUGS_FIXED.md](docs/BUGS_FIXED.md))

1. **Vendor Client Crash on 503/404/Timeout:** Replaced naive `raise_for_status()` in `src/vendor_client.py` with structured error dictionaries.
2. **Missing Department Budget Crash:** Handled unbudgeted departments ("Go To Market") gracefully in `src/data_access.py`.
3. **Approvals Cleared on Missing Department:** Fixed a guardrail defect where missing-department escalations inadvertently cleared financial approvals.
4. **Gemini 3 `thought_signature` Error:** Resolved a 400 Bad Request error on multi-turn function calling where Gemini 3 thought signatures were omitted from response parts. This was discovered during the first real smoke test and was silently hidden because the guardrail fell back to the deterministic decision, so every case still showed PASS. This discovery led to adding the `llm_fallback` column to the eval CSVs and strict scoring where a fallback never counts as a pass for single or staged architectures.

---

## 9. Evaluation & Benchmark Results

### Evaluation Methodology & Public Eval Honesty

- **Dataset:** 21 benchmark cases (10 starter cases + 11 synthetic edge cases).
- **Model:** `gemini-3.1-flash-lite` (selected for reliable 15 RPM / 500 RPD quotas on free-tier; `gemini-2.5-flash-lite` was restricted to 20 requests/day).
- **Public Evaluation Honesty:** The deterministic score of 21/21 reflects policy rules that were validated and refined against my hand-computed ground truth (`evals/ground_truth.json`). Therefore, this 100% score represents a rigorous verification of **rule-and-ground-truth consistency**, rather than independent proof of general AI generalization.
- **Scoring:** Strict evaluation requires exact category match, exact approval set, exact risk flag set, at least one evidence item, grounded risk flags, and zero LLM fallbacks.

### Benchmark Results Table (Source: `evals/results/summary.csv`)

| Architecture | Total Cases | Strict Pass Rate | Category Accuracy | Approvals Exact | LLM Fallbacks | Mean LLM Calls | Mean Latency | Median Latency | p95 Latency | Total Tool Calls |
|---|---|---|---|---|---|---|---|---|---|---|
| **Deterministic Floor** | 21 | **100.0%** (21/21) | **100.0%** (21/21) | **100.0%** (21/21) | n/a | 0.0 | 1.6 ms | 0.9 ms | 4.7 ms | 0 |
| **Architecture A (Single Agent)** | 21 | **90.5%** (19/21) | **100.0%** (21/21) | **95.2%** (20/21) | 0 | 3.0 | 20.4 s | 20.1 s | 24.7 s | 104 |
| **Architecture B (Staged / 2-Agent)** | 21 | **90.5%** (19/21) | **100.0%** (21/21) | **95.2%** (20/21) | 0 | 4.0 | 30.0 s | 26.8 s | 40.9 s | 102 |

Note: In deterministic runs, `llm_fallback_count` in raw CSV is 21 due to `apply_guardrails(draft=None, ...)` tagging the string, but `evals/run_evals.py` explicitly exempts deterministic mode from strict fallback penalization because no LLM was requested.

### Failure Analysis

Across all 42 LLM evaluation runs (21 single, 21 staged), zero mandatory approvals or policy rules were missed. All 4 strict failures were conservative over-escalations:
- **Architecture A (2 over-escalations):**
  - `BASE-01`: LLM added `Procurement` approval on an $800 request (policy floor required only `Manager`).
  - `BASE-06`: LLM added `security_review_required` risk flag in response to prompt injection text in justification.
- **Architecture B (2 over-escalations):**
  - `BASE-06`: Staged reviewer added both `privacy_review_required` and `security_review_required` on the adversarial request.
  - `SYN-02`: Staged reviewer added `Privacy` approval and `privacy_review_required` flag on an expanded engineering tier.

**The Union Guardrail Trade-off:** The post-LLM guardrail unions LLM caution with the deterministic floor. Clamping outputs strictly to the floor would produce 21/21 strict pass on benchmarks, but would discard valid LLM reasoning in ambiguous real-world scenarios.

---

## 10. Architecture Comparison & Ship Decision

### Decision: Ship Architecture A (Single Agent)

1. **Equal Quality & Zero Fallbacks:** Both architectures achieved identical strict pass rates (19/21, 90.5%) and category accuracy (21/21, 100%), with zero fallbacks across all runs.
2. **Lower Latency & Cost:** Architecture A requires 25% fewer LLM turns (3.0 vs 4.0 mean calls) and executes ~32% faster (20.4 s vs 30.0 s mean latency; 24.7 s vs 40.9 s p95).
3. **No Precision Gain from Staging:** While Architecture A over-escalated on `BASE-01` (adding Procurement) and Architecture B's reviewer matched `BASE-01`, Architecture B conversely added unneeded `Privacy` approval and flags on `SYN-02`. Neither architecture demonstrated superior precision.
4. **Honest Appraisal of LLM Role:** The deterministic Python engine alone achieved 21/21 strict pass in under 2 ms. The LLM's primary enterprise value is natural-language synthesis for human approvers, ambiguity resolution, and `use_existing_tool` judgment. Adding multi-agent orchestration overhead provides no measurable benefit for this workflow.

See [Architecture Decision Memo](templates/architecture_decision.md) for the complete decision brief.

---

## 11. Known Limitations ([docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md))

1. **Regex Injection Scanner:** Static pattern heuristics can be bypassed by novel or obfuscated jailbreaks; production systems require semantic classifiers. Heuristic patterns can also false-positive on legitimate procurement language (e.g., "pre-approved", "disregard").
2. **Static Catalog Matching:** Catalog matching uses substring containment and lacks live SaaS license utilization metrics.
3. **Keyword Over-Triggering:** Keyword substrings like `"cloud"` or `"production"` in requested integrations can trigger Security review on standard SaaS configurations.
4. **Eval Runner Cache Flag:** The `--no-cache` CLI argument in `evals/run_evals.py` is not wired (caching is off by default in code; keep `LLM_CACHE` unset).
5. **Tool-Loop Caching:** The first turn of a tool loop could technically be cached without thought signatures if caching were enabled for tool loops, but it is intentionally left off to avoid stale state.
6. **Mock External API:** The mock vendor-risk service has case-sensitive endpoints and operates in-process.
7. **Evaluation Scope:** Benchmark covers 21 synthetic cases on a single model (`gemini-3.1-flash-lite`).
8. **No Variance Estimate:** Each LLM evaluation result is from a single run per case (n=1) with no variance estimate; the 19/21 vs 19/21 tie between architectures should be read as "no measurable difference in this sample" rather than proof of equivalence.
