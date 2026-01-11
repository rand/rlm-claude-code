#!/usr/bin/env python3
"""
Externalize context before compaction.

Called by: hooks/hooks.json PreCompact

This is critical - when Claude Code compacts the conversation,
we need to externalize all context to files so the RLM can
still access it after compaction.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def externalize_context():
    """
    Externalize full context before compaction.

    Saves:
    - Full conversation history
    - All cached files
    - All tool outputs
    - Working memory state
    """
    try:
        from src.state_persistence import get_persistence
        from src.context_manager import (
            externalize_conversation,
            externalize_files,
            externalize_tool_outputs,
        )

        persistence = get_persistence()

        # Get session ID from environment
        session_id = os.environ.get("CLAUDE_SESSION_ID", "default")

        # Ensure session is initialized
        if persistence.current_state is None:
            persistence.init_session(session_id)

        # Create externalization directory
        extern_dir = Path.home() / ".claude" / "rlm-externalized" / session_id
        extern_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())

        # Save current state
        state_file = persistence.save_state()

        # Export context using context_manager functions
        if persistence.current_context:
            # Export conversation
            conv_file = extern_dir / f"conversation_{timestamp}.json"
            conv_data = externalize_conversation(persistence.current_context.messages)
            with open(conv_file, "w") as f:
                json.dump(conv_data, f, indent=2)

            # Export files
            if persistence.current_context.files:
                files_file = extern_dir / f"files_{timestamp}.json"
                files_data = externalize_files(persistence.current_context.files)
                with open(files_file, "w") as f:
                    json.dump(files_data, f, indent=2)

            # Export tool outputs
            if persistence.current_context.tool_outputs:
                outputs_file = extern_dir / f"tool_outputs_{timestamp}.json"
                outputs_data = externalize_tool_outputs(persistence.current_context.tool_outputs)
                with open(outputs_file, "w") as f:
                    json.dump(outputs_data, f, indent=2)

        # Create manifest
        manifest = {
            "session_id": session_id,
            "timestamp": timestamp,
            "state_file": str(state_file),
            "externalized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "context_stats": {
                "messages": len(persistence.current_context.messages) if persistence.current_context else 0,
                "files": len(persistence.current_context.files) if persistence.current_context else 0,
                "tool_outputs": len(persistence.current_context.tool_outputs) if persistence.current_context else 0,
            },
        }

        manifest_file = extern_dir / f"manifest_{timestamp}.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        # Update working memory with externalization info
        persistence.update_working_memory(
            "last_externalization",
            {
                "timestamp": timestamp,
                "manifest": str(manifest_file),
            },
        )

        # Output success
        result = {
            "status": "externalized",
            "session_id": session_id,
            "manifest": str(manifest_file),
            "stats": manifest["context_stats"],
        }
        print(json.dumps(result))

    except ImportError as e:
        result = {"status": "skipped", "reason": f"import_error: {e}"}
        print(json.dumps(result))
    except Exception as e:
        result = {"status": "error", "reason": str(e)}
        print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    externalize_context()
