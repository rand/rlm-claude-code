#!/usr/bin/env python3
"""
Capture tool output for RLM context after tool execution.

Called by: hooks/hooks.json PostToolUse

This captures the output of tools (bash, edit, read) and adds
them to the RLM context for subsequent processing.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def capture_output():
    """
    Capture tool output and add to RLM context.

    Reads output from environment or stdin and updates
    the RLM state persistence layer.
    """
    try:
        from src.state_persistence import get_persistence

        persistence = get_persistence()

        # Get session ID from environment
        session_id = os.environ.get("CLAUDE_SESSION_ID", "default")

        # Ensure session is initialized
        if persistence.current_state is None:
            persistence.init_session(session_id)

        # Read tool output from environment
        tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
        tool_output = os.environ.get("CLAUDE_TOOL_OUTPUT", "")
        exit_code_str = os.environ.get("CLAUDE_TOOL_EXIT_CODE", "")

        exit_code = int(exit_code_str) if exit_code_str.isdigit() else None

        if tool_name and tool_output:
            # Add tool output to context
            persistence.add_tool_output(
                tool_name=tool_name,
                content=tool_output[:50000],  # Limit size
                exit_code=exit_code,
            )

            # If this was a file read, cache the content
            if tool_name == "Read":
                pending_file = persistence.current_state.working_memory.get(
                    "pending_file_read"
                )
                if pending_file:
                    persistence.add_file_to_cache(pending_file, tool_output)
                    # Clear pending
                    persistence.update_working_memory("pending_file_read", None)

            # Track errors for complexity detection
            if exit_code and exit_code != 0:
                error_count = persistence.current_state.working_memory.get(
                    "recent_errors", 0
                )
                persistence.update_working_memory("recent_errors", error_count + 1)

        # Save state periodically
        if persistence.current_state.tool_outputs_count % 10 == 0:
            persistence.save_state()

        # Output success
        result = {
            "status": "captured",
            "tool": tool_name,
            "output_size": len(tool_output) if tool_output else 0,
        }
        print(json.dumps(result))

    except ImportError as e:
        result = {"status": "skipped", "reason": f"import_error: {e}"}
        print(json.dumps(result))
    except Exception as e:
        result = {"status": "error", "reason": str(e)}
        print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    capture_output()
