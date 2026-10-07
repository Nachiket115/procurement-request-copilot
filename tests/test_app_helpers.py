from __future__ import annotations

import unittest

from src.contracts import EvidenceItem, ProcurementDecision, RecommendationCategory, RunTelemetry
from src.ui_helpers import (
    badge_text,
    flag_color,
    format_badge_text,
    format_telemetry,
    group_evidence,
    is_fallback,
)


class TestAppHelpers(unittest.TestCase):
    def test_flag_color_red_categories(self) -> None:
        red_flags = [
            "security_review_required",
            "privacy_review_required",
            "legal_review_required",
            "prompt_injection_detected",
            "vendor_risk_unavailable",
            "SECURITY",
            "injection_attempt",
        ]
        for flag in red_flags:
            self.assertEqual(flag_color(flag), "red", f"Flag {flag} should be red")

    def test_flag_color_orange_categories(self) -> None:
        orange_flags = [
            "budget_insufficient",
            "existing_tool_overlap",
            "vendor_review_expired",
            "conflicting_vendor_evidence",
            "BUDGET_OVER_LIMIT",
        ]
        for flag in orange_flags:
            self.assertEqual(flag_color(flag), "orange", f"Flag {flag} should be orange")

    def test_flag_color_grey_categories(self) -> None:
        grey_flags = [
            "missing_information",
            "missing_field",
            "some_unknown_flag",
            "",
            None,
        ]
        for flag in grey_flags:
            self.assertEqual(flag_color(flag), "grey", f"Flag {flag} should be grey")

    def test_badge_text_formatting(self) -> None:
        self.assertEqual(badge_text("security_review_required"), "Security Review Required")
        self.assertEqual(badge_text("prompt_injection_detected"), "Prompt Injection Detected")
        self.assertEqual(badge_text("budget_insufficient"), "Budget Insufficient")
        self.assertEqual(
            badge_text(RecommendationCategory.route_for_governance_review),
            "Route For Governance Review",
        )
        self.assertEqual(badge_text(""), "")
        self.assertEqual(badge_text(None), "")
        self.assertEqual(format_badge_text("needs_more_information"), "Needs More Information")

    def test_group_evidence_with_pydantic_items(self) -> None:
        items = [
            EvidenceItem(source="software_catalog", finding="PixelCraft found in catalog", reference="catalog.csv"),
            EvidenceItem(source="vendor_risk_api", finding="Status 200 OK", reference="/vendor-risk"),
            EvidenceItem(source="software_catalog", finding="BrandBoard competitor", reference="catalog.csv"),
        ]
        grouped = group_evidence(items)
        self.assertEqual(set(grouped.keys()), {"software_catalog", "vendor_risk_api"})
        self.assertEqual(len(grouped["software_catalog"]), 2)
        self.assertEqual(len(grouped["vendor_risk_api"]), 1)

        # Attribute access & key access
        first = grouped["software_catalog"][0]
        self.assertEqual(first.finding, "PixelCraft found in catalog")
        self.assertEqual(first["finding"], "PixelCraft found in catalog")
        self.assertEqual(first.reference, "catalog.csv")
        self.assertEqual(first["reference"], "catalog.csv")

    def test_group_evidence_with_dicts_and_empty(self) -> None:
        items = [
            {"source": "department_budgets", "finding": "Sufficient budget", "reference": "budgets.csv"},
        ]
        grouped = group_evidence(items)
        self.assertIn("department_budgets", grouped)
        self.assertEqual(grouped["department_budgets"][0]["finding"], "Sufficient budget")

        self.assertEqual(group_evidence([]), {})
        self.assertEqual(group_evidence(None), {})

    def test_format_telemetry_with_decision(self) -> None:
        tel = RunTelemetry(
            llm_calls=2,
            tool_calls=3,
            tool_names=["get_vendor_risk", "get_catalog"],
            latency_ms=1250.5,
            guardrail_corrections=1,
            model_name="gemini-2.5-flash",
        )
        decision = ProcurementDecision(
            request_id="REQ-1001",
            recommendation="Approve request",
            recommendation_category=RecommendationCategory.route_for_standard_review,
            required_approvals=["Manager"],
            next_step="Route to manager",
            telemetry=tel,
        )
        res = format_telemetry(decision)
        self.assertEqual(res["llm_calls"], 2)
        self.assertEqual(res["tool_calls"], 3)
        self.assertEqual(res["tool_names"], ["get_vendor_risk", "get_catalog"])
        self.assertEqual(res["latency_ms"], 1250.5)
        self.assertEqual(res["guardrail_corrections"], 1)
        self.assertEqual(res["model"], "gemini-2.5-flash")
        self.assertEqual(res["architecture"], "single")

    def test_format_telemetry_deterministic_and_none(self) -> None:
        decision_det = ProcurementDecision(
            request_id="REQ-1002",
            recommendation="Approve request",
            next_step="Route to manager",
            telemetry=RunTelemetry(
                llm_calls=0,
                tool_calls=1,
                tool_names=["get_vendor_risk_prefetch"],
                latency_ms=12.0,
                model_name="deterministic",
            ),
        )
        res_det = format_telemetry(decision_det)
        self.assertEqual(res_det["architecture"], "deterministic")
        self.assertEqual(res_det["llm_calls"], 0)
        self.assertEqual(res_det["model"], "deterministic")

        res_none = format_telemetry(None)
        self.assertEqual(res_none["architecture"], "unknown")
        self.assertEqual(res_none["llm_calls"], 0)

    def test_is_fallback(self) -> None:
        d_fallback = ProcurementDecision(
            request_id="REQ-1003",
            recommendation="Deterministic recommendation",
            next_step="Route to manager [llm_fallback_engaged]",
        )
        self.assertTrue(is_fallback(d_fallback))

        d_normal = ProcurementDecision(
            request_id="REQ-1004",
            recommendation="Standard recommendation",
            next_step="Route to manager",
        )
        self.assertFalse(is_fallback(d_normal))

        dict_fallback = {"next_step": "Submit for approval [llm_fallback_engaged]"}
        self.assertTrue(is_fallback(dict_fallback))

        self.assertFalse(is_fallback(None))
        self.assertFalse(is_fallback({}))


if __name__ == "__main__":
    unittest.main()
