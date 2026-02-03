"""Event consumption helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVENTS_DIR = Path.home() / ".claude" / "events"


def read_latest_event(source: str) -> dict[str, Any] | None:
    """Read the most recent event from a source."""
    latest_file = EVENTS_DIR / f"{source}-latest.json"
    if not latest_file.exists():
        return None
    try:
        return json.loads(latest_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def get_dp_phase() -> str:
    """Get current disciplined-process phase."""
    event = read_latest_event("disciplined-process")
    if event and event.get("type") == "phase_transition":
        return event.get("to_phase", "unknown")
    return "unknown"


def get_rlm_mode() -> str:
    """Get current RLM mode."""
    event = read_latest_event("rlm-claude-code")
    if event and event.get("type") == "mode_change":
        return event.get("to_mode", "unknown")
    return "unknown"


def suggested_rlm_mode() -> str:
    """Suggest RLM mode based on DP phase."""
    phase = get_dp_phase()
    mapping = {
        "spec": "thorough",
        "review": "thorough",
        "test": "balanced",
        "implement": "balanced",
        "orient": "balanced",
        "decide": "balanced",
    }
    return mapping.get(phase, "")
