from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from evals.run_evals import run_evaluation_suite, score_case
from src.contracts import EvidenceItem, ProcurementDecision, RecommendationCategory
from src.data_access import get_request
from src.guardrails import apply_guardrails

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = ROOT / "evals" / "ground_truth.json"
FIXTURES_DIR = ROOT / "evals" / "fixtures"


class TestRunEvals(unittest.TestCase):
    def test_score_case_exact_match(self):
        case = {
            "case_id": "TEST-01",
            "recommendation_category": "route_for_standard_review",
            "required_approvals": ["Manager"],
            "risk_flags": ["existing_tool_overlap"],
            "missing_information": [],
            "approvals_must_not_include": ["CFO"],
            "risk_flags_must_not_contain": ["budget_insufficient"],
        }
        decision = ProcurementDecision(
            request_id="REQ-TEST",
            recommendation="Standard review",
            recommendation_category=RecommendationCategory.route_for_standard_review,
            evidence=[EvidenceItem(source="rules_engine", finding="Valid overlap")],
            required_approvals=["Manager"],
            risk_flags=["existing_tool_overlap"],
            missing_information=[],
            next_step="Submit",
            human_review_required=True,
        )
        scores = score_case(case, decision)
        self.assertTrue(scores["category_correct"])
        self.assertTrue(scores["approvals_exact"])
        self.assertTrue(scores["flags_exact"])
        self.assertTrue(scores["strict_pass"])
        self.assertTrue(scores["lenient_pass"])

    def test_score_case_missing_and_extra_approvals(self):
        case = {
            "case_id": "TEST-02",
            "recommendation_category": "route_for_governance_review",
            "required_approvals": ["Department Head", "Procurement", "Security"],
            "risk_flags": ["security_review_required"],
            "missing_information": [],
        }
        # Decision has Department Head and Finance, but missing Procurement & Security
        decision = ProcurementDecision(
            request_id="REQ-TEST-2",
            recommendation="Governance review",
            recommendation_category=RecommendationCategory.route_for_governance_review,
            evidence=[EvidenceItem(source="rules", finding="Finding")],
            required_approvals=["Department Head", "Finance"],
            risk_flags=["security_review_required"],
            missing_information=[],
            next_step="Submit",
            human_review_required=True,
        )
        scores = score_case(case, decision)
        self.assertFalse(scores["approvals_exact"])
        self.assertEqual(scores["approvals_missing"], ["Procurement", "Security"])
        self.assertEqual(scores["approvals_extra"], ["Finance"])
        self.assertFalse(scores["strict_pass"])

    def test_score_case_judgment_items_ignored_in_lenient(self):
        case = {
            "case_id": "TEST-03",
            "recommendation_category": "escalate_to_finance",
            "required_approvals": ["Department Head", "Procurement", "Finance"],
            "risk_flags": ["existing_tool_overlap", "missing_information"],
            "missing_information": ["Department budget record missing"],
            "judgment_approvals": ["Finance"],
            "judgment_risk_flags": ["existing_tool_overlap"],
        }
        # Decision omits Finance and existing_tool_overlap (which are judgment items)
        decision = ProcurementDecision(
            request_id="REQ-TEST-3",
            recommendation="Escalate",
            recommendation_category=RecommendationCategory.escalate_to_finance,
            evidence=[EvidenceItem(source="rules", finding="Finding")],
            required_approvals=["Department Head", "Procurement"],
            risk_flags=["missing_information"],
            missing_information=["Department budget record missing"],
            next_step="Submit",
            human_review_required=True,
        )
        scores = score_case(case, decision)
        self.assertFalse(scores["strict_pass"])
        self.assertTrue(scores["lenient_pass"])

    def test_score_case_must_not_include_violation(self):
        case = {
            "case_id": "TEST-04",
            "recommendation_category": "route_for_standard_review",
            "required_approvals": ["Manager"],
            "risk_flags": [],
            "missing_information": [],
            "approvals_must_not_include": ["CFO", "Finance"],
        }
        decision = ProcurementDecision(
            request_id="REQ-TEST-4",
            recommendation="Review",
            recommendation_category=RecommendationCategory.route_for_standard_review,
            evidence=[EvidenceItem(source="rules", finding="Finding")],
            required_approvals=["Manager", "CFO"],
            risk_flags=[],
            missing_information=[],
            next_step="Submit",
            human_review_required=True,
        )
        scores = score_case(case, decision)
        self.assertFalse(scores["must_not_include_ok"])
        self.assertFalse(scores["strict_pass"])

    def test_deterministic_architecture_all_21_cases_strict_pass(self):
        # Full deterministic architecture gives strict_pass for all 21 cases
        all_cases = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(all_cases), 21)

        fixture_vendors = json.loads((FIXTURES_DIR / "vendors.json").read_text(encoding="utf-8"))
        base_vendors = json.loads((ROOT / "data" / "vendor_risk.json").read_text(encoding="utf-8"))

        for case in all_cases:
            case_id = case["case_id"]
            req_id = case["request_id"]
            is_syn = case_id.startswith("SYN-")
            fix_dir = FIXTURES_DIR if is_syn else None

            req = get_request(req_id, fixtures_dir=fix_dir)
            vendor_name = req.get("vendor_name")

            if vendor_name == "NimbusAI":
                vendor_api = {
                    "status": "unavailable",
                    "vendor_name": "NimbusAI",
                    "error": "Upstream vendor assessment provider is temporarily unavailable.",
                }
            elif vendor_name in fixture_vendors:
                v_data = fixture_vendors[vendor_name]
                vendor_api = {"status": "ok", "vendor_name": vendor_name, **v_data}
            elif vendor_name in base_vendors:
                v_data = base_vendors[vendor_name]
                vendor_api = {"status": "ok", "vendor_name": vendor_name, **v_data}
            else:
                vendor_api = None

            decision = apply_guardrails(
                None,
                req,
                vendor_api_result=vendor_api,
                fixtures_dir=fix_dir,
            )

            scores = score_case(case, decision)
            self.assertTrue(
                scores["strict_pass"],
                f"Case {case_id} failed strict evaluation: {scores}",
            )


    def test_score_case_llm_fallback_affects_single_not_deterministic(self):
        case = {
            "case_id": "TEST-FB",
            "recommendation_category": "route_for_standard_review",
            "required_approvals": ["Manager"],
            "risk_flags": ["existing_tool_overlap"],
            "missing_information": [],
        }
        decision_fallback = ProcurementDecision(
            request_id="REQ-TEST-FB",
            recommendation="Standard review",
            recommendation_category=RecommendationCategory.route_for_standard_review,
            evidence=[EvidenceItem(source="rules_engine", finding="Valid overlap")],
            required_approvals=["Manager"],
            risk_flags=["existing_tool_overlap"],
            missing_information=[],
            next_step="Submit request [llm_fallback_engaged]",
            human_review_required=True,
        )

        # For single architecture: fallback engaged causes failure
        scores_single = score_case(case, decision_fallback, architecture="single")
        self.assertTrue(scores_single["llm_fallback"])
        self.assertFalse(scores_single["strict_pass"])
        self.assertFalse(scores_single["lenient_pass"])

        # For staged architecture: fallback engaged causes failure
        scores_staged = score_case(case, decision_fallback, architecture="staged")
        self.assertTrue(scores_staged["llm_fallback"])
        self.assertFalse(scores_staged["strict_pass"])
        self.assertFalse(scores_staged["lenient_pass"])

        # For deterministic architecture: unaffected by fallback string
        scores_det = score_case(case, decision_fallback, architecture="deterministic")
        self.assertTrue(scores_det["llm_fallback"])
        self.assertTrue(scores_det["strict_pass"])
        self.assertTrue(scores_det["lenient_pass"])


if __name__ == "__main__":
    unittest.main()
