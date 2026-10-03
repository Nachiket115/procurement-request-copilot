from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.data_access import get_request
from src.rules import evaluate_request

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = ROOT / "evals" / "ground_truth.json"
BASE_VENDOR_RISK_PATH = ROOT / "data" / "vendor_risk.json"
FIXTURES_DIR = ROOT / "evals" / "fixtures"
FIXTURES_VENDORS_JSON = FIXTURES_DIR / "vendors.json"


class GroundTruthFixturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
        cls.base_vendor_risk = json.loads(BASE_VENDOR_RISK_PATH.read_text(encoding="utf-8"))
        cls.fixture_vendor_risk = json.loads(FIXTURES_VENDORS_JSON.read_text(encoding="utf-8"))

    def test_21_ground_truth_cases_with_fixtures(self):
        mismatches = []

        for case in self.ground_truth:
            case_id = case["case_id"]
            req_id = case["request_id"]

            with self.subTest(case_id=case_id):
                # 1. Load request
                if case_id.startswith("SYN-"):
                    req = get_request(req_id, fixtures_dir=FIXTURES_DIR)
                else:
                    req = get_request(req_id, fixtures_dir=FIXTURES_DIR)

                vendor_name = req.get("vendor_name")

                # 2. Build vendor API result
                if vendor_name == "NimbusAI":
                    vendor_api = {
                        "status": "unavailable",
                        "vendor_name": "NimbusAI",
                        "error": "Upstream vendor assessment provider is temporarily unavailable.",
                    }
                elif vendor_name in self.fixture_vendor_risk:
                    v_data = self.fixture_vendor_risk[vendor_name]
                    vendor_api = {"status": "ok", "vendor_name": vendor_name, **v_data}
                elif vendor_name in self.base_vendor_risk:
                    v_data = self.base_vendor_risk[vendor_name]
                    vendor_api = {"status": "ok", "vendor_name": vendor_name, **v_data}
                else:
                    vendor_api = None

                # 3. Evaluate request with fixtures_dir
                eval_res = evaluate_request(
                    req,
                    vendor_api_result=vendor_api,
                    fixtures_dir=FIXTURES_DIR,
                )

                # Check category
                cat_match = eval_res["recommendation_category_floor"] == case["recommendation_category"]
                if not cat_match:
                    mismatches.append(
                        f"[{case_id}] Category mismatch: got {eval_res['recommendation_category_floor']!r} vs expected {case['recommendation_category']!r}"
                    )

                # Check required approvals
                appr_match = eval_res["required_approvals"] == case["required_approvals"]
                if not appr_match:
                    mismatches.append(
                        f"[{case_id}] Approvals mismatch: got {eval_res['required_approvals']!r} vs expected {case['required_approvals']!r}"
                    )

                # Check approvals_must_not_include
                for excluded in case.get("approvals_must_not_include", []):
                    if excluded in eval_res["required_approvals"]:
                        mismatches.append(
                            f"[{case_id}] Approvals must not include {excluded!r}, but found in {eval_res['required_approvals']!r}"
                        )

                # Check risk flags
                flags_match = set(eval_res["risk_flags"]) == set(case["risk_flags"])
                if not flags_match:
                    mismatches.append(
                        f"[{case_id}] Risk flags mismatch: got {eval_res['risk_flags']!r} vs expected {case['risk_flags']!r}"
                    )

                # Check risk_flags_must_not_contain
                for excluded_flag in case.get("risk_flags_must_not_contain", []):
                    if excluded_flag in eval_res["risk_flags"]:
                        mismatches.append(
                            f"[{case_id}] Risk flags must not contain {excluded_flag!r}, but found in {eval_res['risk_flags']!r}"
                        )

                # Check missing information
                missing_match = set(eval_res["missing_information"]) == set(case["missing_information"])
                if not missing_match:
                    mismatches.append(
                        f"[{case_id}] Missing info mismatch: got {eval_res['missing_information']!r} vs expected {case['missing_information']!r}"
                    )

                self.assertEqual(
                    eval_res["recommendation_category_floor"],
                    case["recommendation_category"],
                    f"Category mismatch in {case_id}",
                )
                self.assertEqual(
                    eval_res["required_approvals"],
                    case["required_approvals"],
                    f"Approvals mismatch in {case_id}",
                )
                for excluded in case.get("approvals_must_not_include", []):
                    self.assertNotIn(
                        excluded,
                        eval_res["required_approvals"],
                        f"{excluded} should not be in required_approvals for {case_id}",
                    )
                self.assertEqual(
                    set(eval_res["risk_flags"]),
                    set(case["risk_flags"]),
                    f"Risk flags mismatch in {case_id}",
                )
                for excluded_flag in case.get("risk_flags_must_not_contain", []):
                    self.assertNotIn(
                        excluded_flag,
                        eval_res["risk_flags"],
                        f"{excluded_flag} should not be in risk_flags for {case_id}",
                    )
                self.assertEqual(
                    set(eval_res["missing_information"]),
                    set(case["missing_information"]),
                    f"Missing info mismatch in {case_id}",
                )


if __name__ == "__main__":
    unittest.main()
