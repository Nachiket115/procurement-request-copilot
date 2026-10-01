from __future__ import annotations

import logging
import os
from urllib.parse import quote
import requests


def get_vendor_risk(vendor_name: str, timeout_seconds: float = 3.0) -> dict:
    """Low-level API client for vendor risk service."""
    base_url = os.getenv("VENDOR_RISK_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
    url = f"{base_url}/vendor-risk/{quote(vendor_name, safe='')}"
    try:
        response = requests.get(url, timeout=timeout_seconds)
        if response.status_code == 200:
            try:
                data = response.json()
            except Exception:
                return {
                    "status": "unavailable",
                    "vendor_name": vendor_name,
                    "error": "Invalid response body",
                }
            if isinstance(data, dict):
                data.setdefault("status", "ok")
                data.setdefault("vendor_name", vendor_name)
                return data
            return {
                "status": "unavailable",
                "vendor_name": vendor_name,
                "error": "Invalid response body",
            }

        detail = None
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("error_message") or body.get("error")
        except Exception:
            detail = response.text

        if response.status_code == 404:
            return {
                "status": "not_found",
                "vendor_name": vendor_name,
                "error": str(detail) if detail is not None else f"No vendor-risk record for '{vendor_name}'",
            }

        return {
            "status": "unavailable",
            "vendor_name": vendor_name,
            "error": str(detail) if detail is not None else f"Vendor-risk service error ({response.status_code})",
        }
    except requests.exceptions.Timeout as e:
        return {
            "status": "unavailable",
            "vendor_name": vendor_name,
            "error": f"Request timed out: {e}",
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "unavailable",
            "vendor_name": vendor_name,
            "error": f"Connection error: {e}",
        }
    except Exception as e:
        logging.exception("Unexpected error fetching vendor risk for %s: %s", vendor_name, e)
        return {
            "status": "unavailable",
            "vendor_name": vendor_name,
            "error": f"Unexpected error: {e}",
        }

