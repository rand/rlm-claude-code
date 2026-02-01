#!/usr/bin/env python3
"""
Check if RLM should activate based on task complexity.

Called by: hooks/hooks.json UserPromptSubmit

Output Format (Claude Code compliant):
- If RLM activates: outputs hookSpecificOutput with additionalContext
- If RLM does not activate: outputs nothing (preserves prompt suggestions)

The activation decision is also stored in ~/.claude/rlm-state/activation.json
for other hooks to read.
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_state_dir() -> Path:
    """Get RLM state directory, creating if needed."""
    state_dir = Path.home() / ".claude" / "rlm-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def save_activation_state(should_activate: bool, reason: str) -> None:
    """
    Save activation decision to state file for other hooks.

    This allows sync_context.py and other hooks to know the
    current activation state without re-running classification.
    """
    state_file = get_state_dir() / "activation.json"
    state = {
        "activate_rlm": should_activate,
        "reason": reason,
    }
    with open(state_file, "w") as f:
        json.dump(state, f)


def check_complexity(prompt: str) -> None:
    """
    Check complexity and output activation decision.

    This is called on every user prompt to determine if
    RLM mode should activate.

    Output follows Claude Code UserPromptSubmit hook format:
    - Outputs hookSpecificOutput.additionalContext when RLM activates
    - Outputs nothing when RLM does not activate (preserves prompt suggestions)
    """
    should_activate = False
    reason = "unknown"

    try:
        from src.complexity_classifier import should_activate_rlm
        from src.types import SessionContext

        # Create minimal context (full context comes from Claude Code)
        context = SessionContext()

        should_activate, reason = should_activate_rlm(prompt, context)

    except ImportError:
        # Fallback if modules not available
        should_activate = False
        reason = "modules_not_loaded"

    # Always save state for other hooks to read
    save_activation_state(should_activate, reason)

    # Output Claude Code compliant format
    # Only output when RLM activates - otherwise stay silent
    # to avoid interfering with prompt suggestions
    if should_activate:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": f"[RLM auto-activated: {reason}]",
            }
        }
        print(json.dumps(output))
    # When not activating, output nothing - this is important!
    # Non-standard JSON output breaks Claude Code's prompt suggestions


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = sys.stdin.read()

    check_complexity(prompt)
