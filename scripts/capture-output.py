#!/usr/bin/env python3
"""
Capture tool output for RLM context after tool execution.

Called by: hooks/hooks.json PostToolUse

This captures the output of tools (bash, edit, read) and adds
them to the RLM context for subsequent processing.

Input: JSON via stdin with fields:
  - tool_name: Name of the tool that was executed
  - tool_input: Input parameters to the tool
  - tool_output: Output from the tool execution
  - exit_code: Exit code if available
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

    Reads output from stdin JSON and updates the RLM state persistence layer.
    """
    try:
        from src.state_persistence import get_persistence

        persistence = get_persistence()

        # Get session ID from environment
        session_id = os.environ.get("CLAUDE_SESSION_ID", "default")

        # Ensure session is initialized
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
        tool_output = input_data.get("tool_output", os.environ.get("CLAUDE_TOOL_OUTPUT", ""))
        exit_code = input_data.get("exit_code")

        # Handle exit_code which might be string from env
        if exit_code is None:
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

        # Save state periodically (first 5 outputs, then every 10)
        if persistence.current_state:
            count = persistence.current_state.tool_outputs_count
            if count <= 5 or count % 10 == 0:
                persistence.save_state(session_id)

        # Output success (exit 0 to not block tool execution)
        result = {
            "status": "captured",
            "tool": tool_name,
            "output_size": len(tool_output) if tool_output else 0,
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
    capture_output()
