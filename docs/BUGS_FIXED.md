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

## 2. Input Hardening (Not Starter Defects)

The following defensive hardening measures were introduced to ensure resilience across synthetic and edge-case evaluations:

1. **Vendor Name Normalization:**
   - Case-insensitive, whitespace-trimmed matching for the registry lookup in `rules.py` only.
2. **NaN-Safe Registry Field Parsing:**
   - Implemented `clean_str()` and `clean_bool()` helpers to handle `None` and `np.nan` values in pandas DataFrames without string operation crashes. Missing `legal_terms_status` values are safely treated as unknown/non-approved.
3. **Withholding Injected Text from Evidence:**
   - When prompt injection heuristics trigger on external vendor API notes, raw injection payloads are excluded from evidence text findings (displaying `"notes withheld: injection pattern detected"` instead) to prevent downstream model poisoning.
