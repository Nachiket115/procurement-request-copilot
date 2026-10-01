# Known Limitations

This document identifies known technical boundaries, heuristic trade-offs, and operational limitations of the current procurement copilot implementation.

---

## 1. Prompt Injection Scanning Heuristics
- **Pattern Matching Scope:** Injection detection relies on static regex patterns (e.g., `"ignore all procurement rules"`, `"treat this (request )?as cfo-approved"`, `"system override"`). Sophisticated multi-step, obfuscated, or multilingual adversarial jailbreaks may evade pattern detection.
- **Unscanned Free-Text Fields:** Injection detection scans `business_justification`, external vendor API notes, internal vendor registry notes, and catalog notes. Other non-standard request free-text attributes (e.g., custom attachment names or miscellaneous metadata) are not currently evaluated by the heuristic scanner.
- **False Positive Risk:** Legitimate business justifications quoting policies or audit overrides (e.g., "per system override procedure #12") may trigger `prompt_injection_detected`.

---

## 2. Catalog & Overlap Matching
- **Substring Match Broadness:** Catalog matching checks for substring containment between requested product names and catalog entries. Broad product titles may produce false-positive overlap flags.
- **"Cloud" Substring Triggers:** Scanning integration strings for substring keywords like `"cloud"` or `"production"` can trigger Security review on non-production SaaS configurations (e.g., "Cloud storage backup integration").
- **Absence of Usage/Utilization Metrics:** The static `software_catalog.csv` records total `licensed_seats` but lacks active telemetry on seat utilization or license reclaimability. The copilot cannot dynamically verify whether existing tool licenses can be reallocated before suggesting new purchases.

---

## 3. External API & Infrastructure Boundaries
- **Mock Vendor Risk Service & Case-Sensitivity:** The copilot integrates with a local FastAPI mock service (`mock_api/app.py`). The mock API's vendor lookup is case-sensitive, and the low-level vendor client does not normalize vendor names prior to calling the endpoint.
- **LLM Free-Tier Rate Limits:** Deployments utilizing Gemini free-tier endpoints are subject to concurrency constraints (e.g., 10 requests per minute). High-throughput batch evaluation runs require client-side throttling or enterprise quota provisioning.

---

## 4. Policy & Evaluation Operational Boundaries
- **Legal Review on Unavailable Risk for New Vendors:** Routing new vendors with an unreachable risk API (`vendor_risk_unavailable`) to Legal review is an operational judgment call for contract terms diligence.
- **Empty Approvals on Missing Information:** When material fields (`vendor_name`, `annual_cost_usd`, `user_count`, `data_access_level`) are missing, `required_approvals` is set to `[]` (review paused), so downstream reviewers see no approval chain until the required information is supplied.
