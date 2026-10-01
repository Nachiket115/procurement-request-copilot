from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
import requests

from src.vendor_client import get_vendor_risk


class TestResilientVendorClient(unittest.TestCase):
    @patch("requests.get")
    def test_200_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vendor_name": "BrandBoard",
            "risk_level": "medium",
            "security_review_status": "not_completed",
        }
        mock_get.return_value = mock_response

        result = get_vendor_risk("BrandBoard")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["vendor_name"], "BrandBoard")
        self.assertEqual(result["risk_level"], "medium")

    @patch("requests.get")
    def test_404_not_found(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "No vendor-risk record for 'UnknownVendor'"}
        mock_get.return_value = mock_response

        result = get_vendor_risk("UnknownVendor")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["vendor_name"], "UnknownVendor")
        self.assertIn("error", result)

    @patch("requests.get")
    def test_503_service_unavailable(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.json.return_value = {
            "detail": "Upstream vendor assessment provider is temporarily unavailable."
        }
        mock_get.return_value = mock_response

        result = get_vendor_risk("NimbusAI")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["vendor_name"], "NimbusAI")
        self.assertIn("error", result)

    @patch("requests.get")
    def test_timeout_returns_unavailable(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out after 3.0s")

        result = get_vendor_risk("SlowVendor")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["vendor_name"], "SlowVendor")
        self.assertIn("Request timed out", result["error"])

    @patch("requests.get")
    def test_connection_error_returns_unavailable(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to establish connection")

        result = get_vendor_risk("UnreachableVendor")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["vendor_name"], "UnreachableVendor")
        self.assertIn("Connection error", result["error"])

    @patch("requests.get")
    def test_request_exception_returns_unavailable(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Generic network error")

        result = get_vendor_risk("BrokenVendor")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["vendor_name"], "BrokenVendor")
        self.assertIn("Connection error", result["error"])


if __name__ == "__main__":
    unittest.main()
