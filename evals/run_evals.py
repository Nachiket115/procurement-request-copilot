from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import ProcurementDecision
from src.data_access import get_request
from src.guardrails import apply_guardrails
from src.rules import evaluate_request
from src.vendor_client import get_vendor_risk

GROUND_TRUTH_PATH = ROOT / "evals" / "ground_truth.json"
FIXTURES_DIR = ROOT / "evals" / "fixtures"
RESULTS_DIR = ROOT / "evals" / "results"


def start_mock_api(port: int = 8002) -> subprocess.Popen:
    """Start local mock API server with fixtures merged on specified port."""
    env = os.environ.copy()
    env["VENDOR_RISK_FIXTURES_PATH"] = str(FIXTURES_DIR / "vendors.json")
    env["FIXTURES_DIR"] = str(FIXTURES_DIR)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "mock_api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    url = f"http://127.0.0.1:{port}/health"
    start_t = time.time()
    while time.time() - start_t < 10.0:
        if proc.poll() is not None:
            raise RuntimeError(f"Mock API server exited with code {proc.returncode}")
        try:
            r = requests.get(url, timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        proc.terminate()
        raise TimeoutError("Timed out waiting for Mock API server to become healthy.")

    return proc


def score_case(
    case: dict[str, Any],
    decision: ProcurementDecision,
    floor_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a decision against ground truth case definition."""
    cat_val = decision.recommendation_category.value if hasattr(decision.recommendation_category, "value") else str(decision.recommendation_category or "")
    expected_cat = case["recommendation_category"]
    category_correct = (cat_val == expected_cat)

    actual_appr = set(decision.required_approvals)
    expected_appr = set(case.get("required_approvals", []))
    approvals_exact = (actual_appr == expected_appr)
    approvals_missing = sorted(list(expected_appr - actual_appr))
    approvals_extra = sorted(list(actual_appr - expected_appr))

    must_not_include = case.get("approvals_must_not_include", [])
    must_not_include_ok = all(a not in actual_appr for a in must_not_include)

    actual_flags = set(decision.risk_flags)
    expected_flags = set(case.get("risk_flags", []))
    flags_exact = (actual_flags == expected_flags)
    flags_missing = sorted(list(expected_flags - actual_flags))
    flags_extra = sorted(list(actual_flags - expected_flags))

    must_not_contain = case.get("risk_flags_must_not_contain", [])
    flags_must_not_contain_ok = all(f not in actual_flags for f in must_not_contain)

    missing_info_ok = (set(decision.missing_information) == set(case.get("missing_information", [])))
    human_review_ok = (decision.human_review_required is True)

    # Grounded: >=1 evidence item and every risk flag is supported by floor evidence / floor flags
    has_evidence = len(decision.evidence) >= 1
    if floor_eval is not None:
        floor_flags_set = set(floor_eval.get("risk_flags", []))
        flags_grounded = all(f in floor_flags_set or f in expected_flags for f in actual_flags)
    else:
        flags_grounded = True
    grounded = has_evidence and flags_grounded

    strict_pass = bool(
        category_correct
        and approvals_exact
        and must_not_include_ok
        and flags_exact
        and flags_must_not_contain_ok
        and missing_info_ok
        and human_review_ok
        and grounded
    )

    # Lenient evaluation ignores items in judgment_approvals and judgment_risk_flags
    judgment_appr = set(case.get("judgment_approvals", []))
    judgment_flags = set(case.get("judgment_risk_flags", []))

    lenient_appr_ok = ((actual_appr - judgment_appr) == (expected_appr - judgment_appr))
    lenient_flags_ok = ((actual_flags - judgment_flags) == (expected_flags - judgment_flags))

    lenient_pass = bool(
        category_correct
        and lenient_appr_ok
        and must_not_include_ok
        and lenient_flags_ok
        and flags_must_not_contain_ok
        and missing_info_ok
        and human_review_ok
        and grounded
    )

    llm_target_hit: bool | None = None
    if "llm_target_recommendation_category" in case:
        llm_target_hit = (cat_val == case["llm_target_recommendation_category"])

    return {
        "category_correct": category_correct,
        "approvals_exact": approvals_exact,
        "approvals_missing": approvals_missing,
        "approvals_extra": approvals_extra,
        "must_not_include_ok": must_not_include_ok,
        "flags_exact": flags_exact,
        "flags_missing": flags_missing,
        "flags_extra": flags_extra,
        "flags_must_not_contain_ok": flags_must_not_contain_ok,
        "missing_info_ok": missing_info_ok,
        "human_review_ok": human_review_ok,
        "grounded": grounded,
        "strict_pass": strict_pass,
        "lenient_pass": lenient_pass,
        "llm_target_hit": llm_target_hit,
    }


def percentile(data: list[float], pct: float) -> float:
    """Compute percentile from sorted list."""
    if not data:
        return 0.0
    k = (len(data) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return d0 + d1


def run_evaluation_suite(
    architecture: str = "deterministic",
    cases_filter: list[str] | None = None,
    no_cache: bool = False,
) -> int:
    """Execute evaluation harness for specified architecture and cases."""
    all_cases: list[dict[str, Any]] = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

    if cases_filter:
        cases_to_run = [c for c in all_cases if c.get("case_id") in cases_filter or c.get("request_id") in cases_filter]
    else:
        cases_to_run = all_cases

    if not cases_to_run:
        print("No matching cases found to evaluate.")
        return 0

    architectures_to_run = ["deterministic", "single", "staged"] if architecture == "all" else [architecture]

    # Start mock API
    port = 8002
    os.environ["VENDOR_RISK_BASE_URL"] = f"http://127.0.0.1:{port}"
    mock_proc = start_mock_api(port=port)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    try:
        for arch in architectures_to_run:
            if arch in ("single", "staged"):
                try:
                    from src.solution import handle_request
                except (ImportError, AttributeError, NotImplementedError) as e:
                    print(f"Architecture '{arch}' not implemented yet ({e}). Skipping.")
                    continue

            print(f"\n=======================================================")
            print(f" Running Evaluation: Architecture = '{arch}' ({len(cases_to_run)} cases)")
            print(f"=======================================================")

            case_results = []
            latencies: list[float] = []
            total_llm_calls = 0
            total_tool_calls = 0

            for case in cases_to_run:
                case_id = case["case_id"]
                req_id = case["request_id"]
                is_syn = case_id.startswith("SYN-")
                fix_dir = FIXTURES_DIR if is_syn else None

                # 1. Load request
                req = get_request(req_id, fixtures_dir=fix_dir)
                vendor_name = req.get("vendor_name")

                # 2. Fetch vendor API record over HTTP
                if vendor_name and str(vendor_name).strip() and str(vendor_name).strip().lower() != "unknown":
                    vendor_api_res = get_vendor_risk(str(vendor_name).strip())
                else:
                    vendor_api_res = None

                start_time = time.time()

                if arch == "deterministic":
                    decision = apply_guardrails(
                        None,
                        req,
                        vendor_api_result=vendor_api_res,
                        started_at=start_time,
                        llm_calls=0,
                        tool_calls=0,
                        tool_names=[],
                        model_name="deterministic",
                        fixtures_dir=fix_dir,
                    )
                else:
                    # Single or staged LLM solution
                    try:
                        from src.solution import handle_request
                        decision = handle_request(req, architecture=arch, vendor_api_result=vendor_api_res, fixtures_dir=fix_dir)
                    except NotImplementedError:
                        print(f"Architecture '{arch}' raised NotImplementedError on case {case_id}.")
                        continue

                # Floor evaluation for grounding check
                floor_eval = evaluate_request(req, vendor_api_result=vendor_api_res, fixtures_dir=fix_dir)

                # Score case
                scores = score_case(case, decision, floor_eval=floor_eval)

                telemetry = decision.telemetry
                lat_ms = telemetry.latency_ms if (telemetry and telemetry.latency_ms is not None) else (time.time() - start_time) * 1000.0
                latencies.append(lat_ms)

                llm_calls_case = telemetry.llm_calls if (telemetry and telemetry.llm_calls is not None) else 0
                tool_calls_case = telemetry.tool_calls if (telemetry and telemetry.tool_calls is not None) else 0
                total_llm_calls += llm_calls_case
                total_tool_calls += tool_calls_case

                corrections = telemetry.guardrail_corrections if telemetry else 0
                additions = telemetry.guardrail_additions if telemetry else 0
                removals = telemetry.guardrail_removals if telemetry else 0
                model_name = telemetry.model_name if telemetry else arch

                ts = datetime.now(timezone.utc).isoformat()

                row = {
                    "timestamp": ts,
                    "architecture": arch,
                    "model_name": model_name or arch,
                    "case_id": case_id,
                    "request_id": req_id,
                    "strict_pass": scores["strict_pass"],
                    "lenient_pass": scores["lenient_pass"],
                    "category_correct": scores["category_correct"],
                    "approvals_exact": scores["approvals_exact"],
                    "flags_exact": scores["flags_exact"],
                    "missing_info_ok": scores["missing_info_ok"],
                    "must_not_include_ok": scores["must_not_include_ok"],
                    "flags_must_not_contain_ok": scores["flags_must_not_contain_ok"],
                    "grounded": scores["grounded"],
                    "llm_target_hit": "" if scores["llm_target_hit"] is None else scores["llm_target_hit"],
                    "approvals_missing": ";".join(scores["approvals_missing"]),
                    "approvals_extra": ";".join(scores["approvals_extra"]),
                    "flags_missing": ";".join(scores["flags_missing"]),
                    "flags_extra": ";".join(scores["flags_extra"]),
                    "latency_ms": round(lat_ms, 2),
                    "llm_calls": llm_calls_case,
                    "tool_calls": tool_calls_case,
                    "guardrail_corrections": corrections,
                    "guardrail_additions": additions,
                    "guardrail_removals": removals,
                }
                case_results.append(row)

                status_str = "PASS" if scores["strict_pass"] else ("LENIENT" if scores["lenient_pass"] else "FAIL")
                print(f"[{case_id:<7}] {status_str:<7} | Category: {scores['category_correct']!s:<5} | Approvals: {scores['approvals_exact']!s:<5} | Flags: {scores['flags_exact']!s:<5} | Latency: {lat_ms:6.1f}ms")

            if not case_results:
                continue

            # Write architecture cases CSV (OVERWRITE)
            arch_cases_csv = RESULTS_DIR / f"{arch}_cases.csv"
            with arch_cases_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(case_results[0].keys()))
                writer.writeheader()
                writer.writerows(case_results)

            # Compute summary
            n_cases = len(case_results)
            strict_count = sum(1 for r in case_results if r["strict_pass"])
            lenient_count = sum(1 for r in case_results if r["lenient_pass"])
            cat_count = sum(1 for r in case_results if r["category_correct"])
            appr_count = sum(1 for r in case_results if r["approvals_exact"])

            sorted_lat = sorted(latencies)
            mean_lat = sum(sorted_lat) / n_cases if n_cases else 0.0
            med_lat = percentile(sorted_lat, 0.50)
            p95_lat = percentile(sorted_lat, 0.95)

            sum_row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "architecture": arch,
                "model_name": case_results[0]["model_name"],
                "total_cases": n_cases,
                "strict_pass_rate": round(strict_count / n_cases, 4) if n_cases else 0.0,
                "lenient_pass_rate": round(lenient_count / n_cases, 4) if n_cases else 0.0,
                "category_accuracy": round(cat_count / n_cases, 4) if n_cases else 0.0,
                "approvals_exact_rate": round(appr_count / n_cases, 4) if n_cases else 0.0,
                "mean_latency_ms": round(mean_lat, 2),
                "median_latency_ms": round(med_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "total_llm_calls": total_llm_calls,
                "total_tool_calls": total_tool_calls,
            }
            summary_rows.append(sum_row)

            print(f"\n--- Summary ({arch}) ---")
            print(f"Strict Pass Rate:    {strict_count}/{n_cases} ({sum_row['strict_pass_rate']*100:.1f}%)")
            print(f"Lenient Pass Rate:   {lenient_count}/{n_cases} ({sum_row['lenient_pass_rate']*100:.1f}%)")
            print(f"Category Accuracy:   {cat_count}/{n_cases} ({sum_row['category_accuracy']*100:.1f}%)")
            print(f"Approvals Exact:     {appr_count}/{n_cases} ({sum_row['approvals_exact_rate']*100:.1f}%)")
            print(f"Mean Latency:        {mean_lat:.2f} ms")
            print(f"Median Latency:      {med_lat:.2f} ms")
            print(f"P95 Latency:         {p95_lat:.2f} ms")

        # Write summary CSV (OVERWRITE)
        if summary_rows:
            summary_csv = RESULTS_DIR / "summary.csv"
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

    finally:
        # Always terminate mock API
        if mock_proc.poll() is None:
            mock_proc.terminate()
            try:
                mock_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                mock_proc.kill()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Procurement Copilot Evaluation Suite")
    parser.add_argument(
        "--architecture",
        choices=["deterministic", "single", "staged", "all"],
        default="deterministic",
        help="Evaluation architecture mode",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated list of case IDs to evaluate (e.g. BASE-01,SYN-03)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass caching mechanisms",
    )

    args = parser.parse_args()
    cases_list = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None

    exit_code = run_evaluation_suite(
        architecture=args.architecture,
        cases_filter=cases_list,
        no_cache=args.no_cache,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
