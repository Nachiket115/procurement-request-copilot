from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]


def _load_data() -> dict:
    base = json.loads((ROOT / "data" / "vendor_risk.json").read_text(encoding="utf-8"))
    fixtures_env = os.environ.get("VENDOR_RISK_FIXTURES_PATH")
    if fixtures_env:
        p = Path(fixtures_env)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        if p.exists() and p.is_file():
            fixtures = json.loads(p.read_text(encoding="utf-8"))
            base.update(fixtures)
    return base


DATA = _load_data()

app = FastAPI(title="FDE Mock Vendor Risk API", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/vendor-risk/{vendor_name}")
def vendor_risk(vendor_name: str) -> dict:
    name = unquote(vendor_name)
    record = DATA.get(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No vendor-risk record for '{name}'")
    if record.get("force_error"):
        raise HTTPException(status_code=503, detail=record.get("error_message", "Vendor-risk service unavailable"))
    return {"vendor_name": name, **record}
