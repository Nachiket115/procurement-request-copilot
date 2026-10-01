from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

from src.contracts import EvidenceItem, ProcurementDecision, RecommendationCategory
from src.data_access import (
    get_department_budget,
    get_request,
    get_requester_department,
    load_budgets,
    load_employees,
    load_purchase_history,
    load_requests,
    load_software_catalog,
    load_vendors,
)


class DataAccessTests(unittest.TestCase):
    def test_caching_and_immutability(self):
        # Load employees and mutate the returned DataFrame
        df1 = load_employees()
        initial_cols = list(df1.columns)
        self.assertIn("department", initial_cols)

        # Mutate df1
        df1.drop(columns=["department"], inplace=True)
        self.assertNotIn("department", df1.columns)

        # Second call must still return full dataframe with department column
        df2 = load_employees()
        self.assertIn("department", df2.columns)

        # Verify caching does not re-read from disk on repeated calls
        with patch("pandas.read_csv", side_effect=AssertionError("Should not re-read from disk")):
            df3 = load_employees()
            self.assertIn("department", df3.columns)

    def test_get_department_budget(self):
        # Standard lookup
        res_eng = get_department_budget("Engineering")
        self.assertTrue(res_eng["found"])
        self.assertEqual(res_eng["department"], "Engineering")
        self.assertEqual(res_eng["available_usd"], 26000)
        self.assertIsInstance(res_eng["annual_software_budget_usd"], int)
        self.assertIsInstance(res_eng["committed_usd"], int)
        self.assertIsInstance(res_eng["available_usd"], int)

        # Whitespace and case insensitivity
        res_eng_space = get_department_budget(" engineering ")
        self.assertTrue(res_eng_space["found"])
        self.assertEqual(res_eng_space["department"], "Engineering")
        self.assertEqual(res_eng_space["available_usd"], 26000)

        # Missing department
        res_gtm = get_department_budget("Go To Market")
        self.assertFalse(res_gtm["found"])
        self.assertEqual(res_gtm["department"], "Go To Market")
        self.assertEqual(res_gtm["error"], "Department budget record missing")

    def test_get_requester_department(self):
        # From requester_id via employees.csv (E004 -> Finance)
        req_1001 = get_request("REQ-1001")
        dept = get_requester_department(req_1001)
        self.assertEqual(dept, "Finance")

        # From request dict containing "department" directly
        custom_req = {"department": "Security", "product_name": "Vault"}
        dept_custom = get_requester_department(custom_req)
        self.assertEqual(dept_custom, "Security")

        # Unknown employee / missing department returns None
        empty_req = {"requester_id": "UNKNOWN_E999"}
        self.assertIsNone(get_requester_department(empty_req))

    def test_get_request(self):
        # Dict passed directly returns as-is
        req_dict = {"request_id": "CUSTOM-1", "product_name": "TestTool"}
        self.assertEqual(get_request(req_dict), req_dict)

        # Valid string ID lookup
        req_found = get_request("REQ-1001")
        self.assertEqual(req_found["request_id"], "REQ-1001")

        # Unknown string ID raises KeyError
        with self.assertRaises(KeyError):
            get_request("REQ-DOES-NOT-EXIST")

    def test_fixtures_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create fixture requests.json and employees.csv
            fixture_requests = [
                {
                    "request_id": "REQ-FIXTURE-1",
                    "requester_id": "EFIX-1",
                    "department": "SpecialOps",
                    "product_name": "FixtureTool",
                }
            ]
            (tmppath / "requests.json").write_text(json.dumps(fixture_requests), encoding="utf-8")

            fixture_employees = pd.DataFrame(
                [
                    {
                        "employee_id": "EFIX-1",
                        "name": "Fixture User",
                        "department": "SpecialOps",
                        "manager_id": "",
                        "level": "IC5",
                        "country": "US",
                    }
                ]
            )
            fixture_employees.to_csv(tmppath / "employees.csv", index=False)

            # 1. Default: fixtures NOT loaded
            default_requests = load_requests()
            self.assertFalse(any(r["request_id"] == "REQ-FIXTURE-1" for r in default_requests))
            with self.assertRaises(KeyError):
                get_request("REQ-FIXTURE-1")

            # 2. Loaded when fixtures_dir is passed explicitly
            explicit_requests = load_requests(fixtures_dir=tmppath)
            self.assertTrue(any(r["request_id"] == "REQ-FIXTURE-1" for r in explicit_requests))
            req_from_fix = get_request("REQ-FIXTURE-1", fixtures_dir=tmppath)
            self.assertEqual(req_from_fix["request_id"], "REQ-FIXTURE-1")

            fix_employees = load_employees(fixtures_dir=tmppath)
            self.assertTrue(any(fix_employees["employee_id"] == "EFIX-1"))

            # 3. Loaded when FIXTURES_DIR env var is set (and restored afterwards)
            old_env = os.environ.get("FIXTURES_DIR")
            try:
                os.environ["FIXTURES_DIR"] = str(tmppath)
                env_requests = load_requests()
                self.assertTrue(any(r["request_id"] == "REQ-FIXTURE-1" for r in env_requests))
            finally:
                if old_env is not None:
                    os.environ["FIXTURES_DIR"] = old_env
                else:
                    os.environ.pop("FIXTURES_DIR", None)

    def test_procurement_decision_backward_compatibility(self):
        # Validates with only the original required fields
        decision = ProcurementDecision(
            request_id="REQ-TEST",
            recommendation="Approve purchase",
            evidence=[EvidenceItem(source="test", finding="Valid test finding")],
            next_step="Send to manager",
        )
        self.assertEqual(decision.request_id, "REQ-TEST")
        self.assertIsNone(decision.recommendation_category)
        self.assertEqual(decision.recommendation, "Approve purchase")

        # Also validates with recommendation_category
        decision_with_cat = ProcurementDecision(
            request_id="REQ-TEST-2",
            recommendation="Route to standard review",
            recommendation_category=RecommendationCategory.route_for_standard_review,
            evidence=[EvidenceItem(source="test", finding="Valid test finding")],
            next_step="Send to manager",
        )
        self.assertEqual(
            decision_with_cat.recommendation_category,
            RecommendationCategory.route_for_standard_review,
        )


if __name__ == "__main__":
    unittest.main()
