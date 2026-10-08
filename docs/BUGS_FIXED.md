# Bugs Fixed

> **Note on Starter Pack Integrity:** All baseline starter checks passed on the unmodified starter pack because starter tests only evaluated happy paths. The defects documented below sit in operational and error-handling failure paths.

---

## 1. Real Defects & Defect Fixes

### 1.1 `src/vendor_client.py`: 503, 404, Timeout, Connection, and Non-JSON Crash
- **File:** `src/vendor_client.py`
- **Root Cause:** `get_vendor_risk()` relied on `response.raise_for_status()`, causing unhandled exceptions (`requests.exceptions.HTTPError`, `requests.exceptions.Timeout`, `requests.exceptions.ConnectionError`, `json.decoder.JSONDecodeError`) when external services suffered outages (503), missing records (404), network timeouts, or returned non-JSON responses.
- **Fix:** Wrapped HTTP communication in robust `try...except` handling. Structured dicts with status indicators are now returned:
  - `404 Not Found` -> `{"status": "not_found", "vendor_name": vendor_name, "error": ...}`
  - `503 Unavailable` / `Timeout` / `ConnectionError` / non-JSON 200 -> `{"status": "unavailable", "vendor_name": vendor_name, "error": ...}`
  - Stringified error messages to guarantee consistent string types.
  - Logged unexpected exceptions with `logging.exception`.
- **Regression Tests:** `tests/test_resilient_vendor_client.py`
  - `test_200_non_json_body_returns_unavailable`
  - `test_404_not_found`
  - `test_503_service_unavailable`
  - `test_timeout_returns_unavailable`
  - `test_connection_error_returns_unavailable`
  - `test_request_exception_returns_unavailable`
  - `test_url_encoding_and_timeout`

---

### 1.2 `src/data_access.py`: Missing Department Budget Handling ("Go To Market")
- **File:** `src/data_access.py`
- **Root Cause:** Starter `data_access.py` only returned raw DataFrames and had no safe department budget lookup helper at all; a naive filter returned an empty DataFrame with no safe fallback for unbudgeted departments like "Go To Market".
- **Fix:** Implemented `get_department_budget(department)` with case-insensitive whitespace-stripped matching:
  - If found: returns `{"found": True, "department": canonical_name, "annual_software_budget_usd": int, "committed_usd": int, "available_usd": int}`.
  - If missing: returns `{"found": False, "department": input_name, "error": "Department budget record missing"}` instead of raising.
- **Regression Tests:** `tests/test_data_access.py`
  - `test_get_department_budget` (tests `"Engineering"`, `" engineering "`, and `"Go To Market"`)

---

### 1.3 `src/guardrails.py`: Approvals Cleared on Missing Department
- **File:** `src/guardrails.py`
- **Root Cause:** When a department was missing from `department_budgets.csv`, the guardrail's missing-department escalation path inadvertently cleared previously computed financial approvals, dropping required approvers from the final decision.
- **Fix:** Ensured that missing-department escalations (adding `Finance` to `required_approvals`) are unioned with the existing approval set rather than replacing it, preserving all previously determined financial and governance approvals.

---

### 1.4 `src/llm.py`: Gemini 3 `thought_signature` Error on Multi-Turn Function Calling
- **File:** `src/llm.py`
- **Root Cause:** Gemini 3 models return `thought_signature` fields in response parts during multi-turn function calling. When these parts were fed back into subsequent turns without the `thought_signature`, the API returned a `400 Bad Request` error. This defect was silently hidden during initial development because the post-LLM guardrail automatically fell back to the deterministic decision on any LLM failure, so every evaluation case still showed PASS.
- **Fix:** Ensured that `thought_signature` fields from Gemini response parts are preserved and included in subsequent multi-turn conversation history.
- **Impact on Evaluation:** This discovery led to adding the `llm_fallback` column to the evaluation CSVs and implementing strict scoring where a fallback never counts as a pass for single-agent or staged architectures.

---

## 2. Input Hardening (Not Starter Defects)

The following defensive hardening measures were introduced to ensure resilience across synthetic and edge-case evaluations:

1. **Vendor Name Normalization:**
   - Case-insensitive, whitespace-trimmed matching for the registry lookup in `rules.py` only.
2. **NaN-Safe Registry Field Parsing:**
   - Implemented `clean_str()` and `clean_bool()` helpers to handle `None` and `np.nan` values in pandas DataFrames without string operation crashes. Missing `legal_terms_status` values are safely treated as unknown/non-approved.
3. **Withholding Injected Text from Evidence:**
   - When prompt injection heuristics trigger on external vendor API notes, raw injection payloads are excluded from evidence text findings (displaying `"notes withheld: injection pattern detected"` instead) to prevent downstream model poisoning.
