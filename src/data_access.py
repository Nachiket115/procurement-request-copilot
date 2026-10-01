from __future__ import annotations

import copy
import functools
import json
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _resolve_fixtures_dir(fixtures_dir: Path | str | None = None) -> str | None:
    if fixtures_dir is not None:
        return str(fixtures_dir)
    env_dir = os.getenv("FIXTURES_DIR")
    if env_dir:
        return str(env_dir)
    return None


@functools.lru_cache(maxsize=None)
def _load_employees_cached(fixtures_key: str | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "employees.csv")
    if fixtures_key:
        fixture_file = Path(fixtures_key) / "employees.csv"
        if fixture_file.is_file():
            fix_df = pd.read_csv(fixture_file)
            df = pd.concat([df, fix_df], ignore_index=True)
    return df


def load_employees(fixtures_dir: Path | str | None = None) -> pd.DataFrame:
    key = _resolve_fixtures_dir(fixtures_dir)
    return _load_employees_cached(key).copy()


@functools.lru_cache(maxsize=None)
def _load_budgets_cached(fixtures_key: str | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "department_budgets.csv")
    if fixtures_key:
        fixture_file = Path(fixtures_key) / "department_budgets.csv"
        if fixture_file.is_file():
            fix_df = pd.read_csv(fixture_file)
            df = pd.concat([df, fix_df], ignore_index=True)
    return df


def load_budgets(fixtures_dir: Path | str | None = None) -> pd.DataFrame:
    key = _resolve_fixtures_dir(fixtures_dir)
    return _load_budgets_cached(key).copy()


@functools.lru_cache(maxsize=None)
def _load_software_catalog_cached(fixtures_key: str | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "software_catalog.csv")
    if fixtures_key:
        fixture_file = Path(fixtures_key) / "software_catalog.csv"
        if fixture_file.is_file():
            fix_df = pd.read_csv(fixture_file)
            df = pd.concat([df, fix_df], ignore_index=True)
    return df


def load_software_catalog(fixtures_dir: Path | str | None = None) -> pd.DataFrame:
    key = _resolve_fixtures_dir(fixtures_dir)
    return _load_software_catalog_cached(key).copy()


@functools.lru_cache(maxsize=None)
def _load_vendors_cached(fixtures_key: str | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "vendors.csv")
    if fixtures_key:
        fixture_file = Path(fixtures_key) / "vendors.csv"
        if fixture_file.is_file():
            fix_df = pd.read_csv(fixture_file)
            df = pd.concat([df, fix_df], ignore_index=True)
    return df


def load_vendors(fixtures_dir: Path | str | None = None) -> pd.DataFrame:
    key = _resolve_fixtures_dir(fixtures_dir)
    return _load_vendors_cached(key).copy()


@functools.lru_cache(maxsize=None)
def _load_purchase_history_cached(fixtures_key: str | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "purchase_history.csv")
    if fixtures_key:
        fixture_file = Path(fixtures_key) / "purchase_history.csv"
        if fixture_file.is_file():
            fix_df = pd.read_csv(fixture_file)
            df = pd.concat([df, fix_df], ignore_index=True)
    return df


def load_purchase_history(fixtures_dir: Path | str | None = None) -> pd.DataFrame:
    key = _resolve_fixtures_dir(fixtures_dir)
    return _load_purchase_history_cached(key).copy()


@functools.lru_cache(maxsize=None)
def _load_requests_cached(fixtures_key: str | None) -> tuple[dict, ...]:
    base_requests = json.loads((DATA_DIR / "requests.json").read_text(encoding="utf-8"))
    if fixtures_key:
        fixture_file = Path(fixtures_key) / "requests.json"
        if fixture_file.is_file():
            fix_requests = json.loads(fixture_file.read_text(encoding="utf-8"))
            base_requests = base_requests + fix_requests
    return tuple(copy.deepcopy(base_requests))


def load_requests(fixtures_dir: Path | str | None = None) -> list[dict]:
    key = _resolve_fixtures_dir(fixtures_dir)
    return [copy.deepcopy(r) for r in _load_requests_cached(key)]


def get_request(request: str | dict, fixtures_dir: Path | str | None = None) -> dict:
    if isinstance(request, dict):
        return request
    if isinstance(request, str):
        for req in load_requests(fixtures_dir=fixtures_dir):
            if req.get("request_id") == request:
                return dict(req)
        raise KeyError(f"Unknown request_id: {request}")
    raise KeyError(f"Invalid request type: {type(request)}")


# A missing department budget returns a structured missing result instead of raising,
# because the procurement copilot must route such requests to Finance rather than crash.
def get_department_budget(department: str, fixtures_dir: Path | str | None = None) -> dict:
    clean_dept = (department or "").strip()
    budgets = load_budgets(fixtures_dir=fixtures_dir)
    clean_dept_lower = clean_dept.lower()
    for _, row in budgets.iterrows():
        dept_name = str(row.get("department", "")).strip()
        if dept_name.lower() == clean_dept_lower:
            return {
                "found": True,
                "department": dept_name,
                "annual_software_budget_usd": int(row["annual_software_budget_usd"]),
                "committed_usd": int(row["committed_usd"]),
                "available_usd": int(row["available_usd"]),
            }
    return {
        "found": False,
        "department": department,
        "error": "Department budget record missing",
    }


def get_requester_department(request: dict, fixtures_dir: Path | str | None = None) -> str | None:
    if not isinstance(request, dict):
        return None
    if "department" in request and request["department"]:
        return str(request["department"])
    requester_id = request.get("requester_id")
    if requester_id:
        employees = load_employees(fixtures_dir=fixtures_dir)
        matches = employees[employees["employee_id"] == str(requester_id)]
        if not matches.empty:
            return str(matches.iloc[0]["department"])
    return None


@functools.lru_cache(maxsize=None)
def _load_policy_text_cached(fixtures_key: str | None) -> str:
    return (DATA_DIR / "procurement_policy.md").read_text(encoding="utf-8")


def load_policy_text(fixtures_dir: Path | str | None = None) -> str:
    key = _resolve_fixtures_dir(fixtures_dir)
    return _load_policy_text_cached(key)
