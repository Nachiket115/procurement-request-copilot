from __future__ import annotations

import importlib
import os
from pathlib import Path
import unittest
from fastapi.testclient import TestClient
import mock_api.app

ROOT = Path(__file__).resolve().parents[1]


class MockApiFixturesTests(unittest.TestCase):
    def test_vendor_risk_fixtures_env(self):
        old_env = os.environ.get("VENDOR_RISK_FIXTURES_PATH")
        try:
            # 1. With env unset, InjectedVendor returns 404
            os.environ.pop("VENDOR_RISK_FIXTURES_PATH", None)
            importlib.reload(mock_api.app)
            client = TestClient(mock_api.app.app)

            r_inj_unset = client.get("/vendor-risk/InjectedVendor")
            self.assertEqual(r_inj_unset.status_code, 404)

            # 2. With env set to evals/fixtures/vendors.json, InjectedVendor returns 200 and BrandBoard returns 200
            os.environ["VENDOR_RISK_FIXTURES_PATH"] = str(ROOT / "evals" / "fixtures" / "vendors.json")
            importlib.reload(mock_api.app)
            client = TestClient(mock_api.app.app)

            r_inj_set = client.get("/vendor-risk/InjectedVendor")
            self.assertEqual(r_inj_set.status_code, 200)
            self.assertEqual(r_inj_set.json()["vendor_name"], "InjectedVendor")

            r_bb_set = client.get("/vendor-risk/BrandBoard")
            self.assertEqual(r_bb_set.status_code, 200)
            self.assertEqual(r_bb_set.json()["vendor_name"], "BrandBoard")

        finally:
            if old_env is not None:
                os.environ["VENDOR_RISK_FIXTURES_PATH"] = old_env
            else:
                os.environ.pop("VENDOR_RISK_FIXTURES_PATH", None)
            importlib.reload(mock_api.app)


if __name__ == "__main__":
    unittest.main()
