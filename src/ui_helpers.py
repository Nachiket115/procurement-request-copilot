from __future__ import annotations

from typing import Any


def flag_color(flag: str | None) -> str:
    """Return badge color ('red', 'orange', or 'grey') based on risk flag category.

    - red: security, privacy, legal, injection, unavailable
    - orange: budget, overlap, expired, conflict
    - grey: missing_information, unknown
    """
    if not flag:
        return "grey"
    clean = str(flag).strip().lower()

    red_keywords = ("security", "privacy", "legal", "injection", "unavailable")
    orange_keywords = ("budget", "overlap", "expired", "conflict")
    grey_keywords = ("missing_information", "missing")

    for kw in red_keywords:
        if kw in clean:
            return "red"
    for kw in orange_keywords:
        if kw in clean:
            return "orange"
    for kw in grey_keywords:
        if kw in clean:
            return "grey"
    return "grey"


def badge_text(text: Any) -> str:
    """Format an enum value, risk flag, or category identifier into a readable badge label."""
    if text is None:
        return ""
    val = text.value if hasattr(text, "value") else str(text)
    val = val.strip()
    if not val:
        return ""
    return val.replace("_", " ").title()


format_badge_text = badge_text


class EvidenceDict(dict):
    """Dictionary supporting attribute-style access for evidence items."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def group_evidence(items: list[Any] | None) -> dict[str, list[EvidenceDict]]:
    """Group evidence items by source, supporting EvidenceItem models and dictionaries."""
    grouped: dict[str, list[EvidenceDict]] = {}
    if not items:
        return grouped

    for item in items:
        if hasattr(item, "model_dump"):
            data = item.model_dump()
        elif hasattr(item, "__dict__"):
            data = dict(item.__dict__)
        elif isinstance(item, dict):
            data = dict(item)
        else:
            continue

        source = str(data.get("source", "other")).strip() or "other"
        if source not in grouped:
            grouped[source] = []

        entry = EvidenceDict({
            "source": source,
            "finding": str(data.get("finding", "")),
            "reference": data.get("reference"),
        })
        grouped[source].append(entry)

    return grouped


def format_telemetry(decision: Any) -> dict[str, Any]:
    """Extract and format execution telemetry from a decision or telemetry object."""
    if decision is None:
        return {
            "architecture": "unknown",
            "model": "unknown",
            "model_name": "unknown",
            "llm_calls": 0,
            "tool_calls": 0,
            "tool_names": [],
            "latency_ms": 0.0,
            "guardrail_corrections": 0,
            "guardrail_additions": 0,
            "guardrail_removals": 0,
        }

    # Extract telemetry container
    if hasattr(decision, "llm_calls") or (isinstance(decision, dict) and "llm_calls" in decision and "telemetry" not in decision):
        telemetry = decision
    elif hasattr(decision, "telemetry"):
        telemetry = decision.telemetry
    elif isinstance(decision, dict):
        telemetry = decision.get("telemetry")
    else:
        telemetry = None

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if hasattr(obj, key):
            val = getattr(obj, key)
            return val if val is not None else default
        if isinstance(obj, dict):
            val = obj.get(key)
            return val if val is not None else default
        return default

    if telemetry is None:
        model = _get(decision, "model_name", "unknown")
        arch = "deterministic" if model == "deterministic" else "unknown"
        return {
            "architecture": arch,
            "model": model,
            "model_name": model,
            "llm_calls": 0,
            "tool_calls": 0,
            "tool_names": [],
            "latency_ms": 0.0,
            "guardrail_corrections": 0,
            "guardrail_additions": 0,
            "guardrail_removals": 0,
        }

    llm_calls = _get(telemetry, "llm_calls", 0)
    tool_calls = _get(telemetry, "tool_calls", 0)
    tool_names = list(_get(telemetry, "tool_names", []))
    latency_ms = _get(telemetry, "latency_ms", 0.0)
    guardrail_corrections = _get(telemetry, "guardrail_corrections", 0)
    guardrail_additions = _get(telemetry, "guardrail_additions", 0)
    guardrail_removals = _get(telemetry, "guardrail_removals", 0)
    model_name = _get(telemetry, "model_name", "unknown")

    arch = _get(decision, "architecture", None)
    if not arch:
        if model_name == "deterministic":
            arch = "deterministic"
        elif model_name and model_name != "unknown":
            arch = "single"
        else:
            arch = "unknown"

    return {
        "architecture": arch,
        "model": model_name,
        "model_name": model_name,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "tool_names": tool_names,
        "latency_ms": float(latency_ms),
        "guardrail_corrections": int(guardrail_corrections),
        "guardrail_additions": int(guardrail_additions),
        "guardrail_removals": int(guardrail_removals),
    }


def is_fallback(decision: Any) -> bool:
    """Return True if decision next_step indicates LLM fallback was engaged."""
    if decision is None:
        return False
    next_step = ""
    if hasattr(decision, "next_step"):
        next_step = decision.next_step or ""
    elif isinstance(decision, dict):
        next_step = decision.get("next_step") or ""
    return "[llm_fallback_engaged]" in str(next_step)
