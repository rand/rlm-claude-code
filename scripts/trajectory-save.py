#!/usr/bin/env python3
"""Trajectory save hook (Python fallback).

Called by hook-dispatch.sh when Go binary is unavailable.
Reads persisted session state and outputs a trajectory summary.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from state_persistence import get_persistence

    session_id = os.environ.get("CLAUDE_SESSION_ID", "default")
    persistence = get_persistence()
    persistence.restore_state(session_id)
    state = persistence.current_state
    summary = {
        "session_id": session_id,
        "trajectory_events": state.trajectory_events_count if state else 0,
        "trajectory_path": state.trajectory_path if state else None,
        "tokens_used": state.total_tokens_used if state else 0,
    }
    print(json.dumps({"status": "saved", "summary": summary}))
except FileNotFoundError:
    print(json.dumps({"status": "skipped", "reason": "no_session_state"}))
except Exception as e:
    print(json.dumps({"status": "skipped", "reason": str(e)}))
