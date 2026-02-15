#!/usr/bin/env python3
"""
Sync tool context with RLM state before tool execution.

Called by: hooks/hooks.json PreToolUse

This ensures the RLM environment has access to the latest
context before any tool (bash, edit, read) is executed.

Input: JSON via stdin with fields:
  - tool_name: Name of the tool to be executed
  - tool_input: Input parameters to the tool
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def sync_context():
    """
    Sync context from Claude Code to RLM state.

    Reads context from stdin JSON and updates the RLM state persistence layer.
    """
    try:
        from src.state_persistence import get_persistence

        persistence = get_persistence()

        # Get session ID from environment
        session_id = os.environ.get("CLAUDE_SESSION_ID", "default")

        # Initialize or restore session
        if persistence.current_state is None:
            persistence.init_session(session_id)

        # Read JSON input from stdin (Claude Code passes data this way)
        input_data = {}
        try:
            input_data = json.load(sys.stdin)
        except json.JSONDecodeError:
            # No valid JSON input, try environment variables as fallback
            pass

        # Extract tool information from stdin JSON or environment
        tool_name = input_data.get("tool_name", os.environ.get("CLAUDE_TOOL_NAME", ""))
        tool_input = input_data.get("tool_input", os.environ.get("CLAUDE_TOOL_INPUT", ""))

        if tool_name:
            # Log tool invocation to working memory
            input_preview = ""
            if tool_input:
                if isinstance(tool_input, str):
                    input_preview = tool_input[:200]
                else:
                    input_preview = json.dumps(tool_input)[:200]

            persistence.update_working_memory(
                "last_tool",
                {"name": tool_name, "input_preview": input_preview},
            )

        # If read tool, prepare to cache file content
        if tool_name == "Read":
            try:
                # tool_input might be a JSON string or already a dict
                if isinstance(tool_input, str):
                    input_json = json.loads(tool_input) if tool_input else {}
                else:
                    input_json = tool_input if tool_input else {}

                file_path = input_json.get("file_path", "")
                if file_path:
                    persistence.update_working_memory("pending_file_read", file_path)
            except json.JSONDecodeError:
                pass

        # Save state to disk so it persists across process boundaries
        persistence.save_state(session_id)

        # Output success (exit 0 to not block tool execution)
        result = {
            "status": "synced",
            "session_id": session_id,
            "rlm_active": persistence.current_state.rlm_active if persistence.current_state else False,
        }
        print(json.dumps(result))

    except ImportError as e:
        # Modules not available - output minimal result and exit cleanly
        result = {"status": "skipped", "reason": f"import_error: {e}"}
        print(json.dumps(result))
    except Exception as e:
        # Log error but don't block tool execution (exit 0)
        result = {"status": "error", "reason": str(e)}
        print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    sync_context()
