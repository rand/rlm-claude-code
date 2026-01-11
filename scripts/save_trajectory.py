#!/usr/bin/env python3
"""
Save trajectory on session end.

Called by: hooks/hooks.json Stop

Exports the full RLM trajectory for analysis and replay.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def save_trajectory():
    """
    Save trajectory and final state on session end.

    Exports:
    - Full trajectory JSON
    - Session summary
    - Cost statistics
    """
    try:
        from src.state_persistence import get_persistence
        from src.config import RLMConfig

        persistence = get_persistence()
        config = RLMConfig.load()

        # Get session ID from environment
        session_id = os.environ.get("CLAUDE_SESSION_ID", "default")

        # Check if we have state to save
        if persistence.current_state is None:
            result = {"status": "skipped", "reason": "no_active_session"}
            print(json.dumps(result))
            return

        # Create trajectories directory from config
        trajectories_dir = Path(config.trajectory.export_path).expanduser()
        trajectories_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        date_str = time.strftime("%Y%m%d")

        # Save final state
        state_file = persistence.save_state()

        # Create session summary
        summary = {
            "session_id": session_id,
            "timestamp": timestamp,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_info": {
                "created_at": persistence.current_state.created_at,
                "ended_at": time.time(),
            },
            "rlm_stats": {
                "was_active": persistence.current_state.rlm_active,
                "max_depth": persistence.current_state.current_depth,
                "total_recursive_calls": persistence.current_state.total_recursive_calls,
                "total_tokens_used": persistence.current_state.total_tokens_used,
                "trajectory_events": persistence.current_state.trajectory_events_count,
            },
            "context_stats": {
                "tool_outputs_count": persistence.current_state.tool_outputs_count,
                "files_cached": len(persistence.current_state.file_cache),
            },
            "state_file": str(state_file),
        }

        # Save summary
        summary_file = trajectories_dir / f"{date_str}_{session_id}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # If trajectory was being recorded, note its location
        if persistence.current_state.trajectory_path:
            summary["trajectory_file"] = persistence.current_state.trajectory_path

        # Clean up old sessions (optional, configurable)
        cleanup_enabled = os.environ.get("RLM_CLEANUP_OLD_SESSIONS", "true").lower() == "true"
        if cleanup_enabled:
            cleaned = persistence.cleanup_old_sessions(max_age_days=7)
            summary["cleaned_old_sessions"] = cleaned

        # Output success
        result = {
            "status": "saved",
            "session_id": session_id,
            "summary_file": str(summary_file),
            "stats": summary["rlm_stats"],
        }
        print(json.dumps(result))

    except ImportError as e:
        result = {"status": "skipped", "reason": f"import_error: {e}"}
        print(json.dumps(result))
    except Exception as e:
        result = {"status": "error", "reason": str(e)}
        print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    save_trajectory()
