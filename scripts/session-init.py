#!/usr/bin/env python3
"""Session initialization hook (Python fallback).

Called by hook-dispatch.sh when Go binary is unavailable.
Initializes RLM session state for cross-hook persistence.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from state_persistence import get_persistence

    session_id = os.environ.get("CLAUDE_SESSION_ID", "default")
    persistence = get_persistence()
    persistence.init_session(session_id)
    persistence.save_state()
    print(json.dumps({"status": "initialized", "session_id": session_id}))
except Exception as e:
    # Fail-open: don't block Claude Code
    print(json.dumps({"status": "skipped", "reason": str(e)}))
