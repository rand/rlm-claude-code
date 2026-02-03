"""Event emission helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENTS_DIR = Path.home() / ".claude" / "events"


def emit_event(event: dict[str, Any], source: str) -> None:
    """Emit an event to the log and latest snapshot."""
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    if "timestamp" not in event:
        event["timestamp"] = datetime.now(UTC).isoformat()
    event.setdefault("source", source)

    # Append to JSONL log
    log_file = EVENTS_DIR / f"{source}-events.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(event) + "\n")

    # Write latest snapshot (indented for readability)
    latest_file = EVENTS_DIR / f"{source}-latest.json"
    latest_file.write_text(json.dumps(event, indent=2) + "\n")
