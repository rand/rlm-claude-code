#!/usr/bin/env python3
"""
Check if RLM should activate based on task complexity.

Called by: hooks/hooks.json UserPromptSubmit
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_complexity(prompt: str):
    """
    Check complexity and output activation decision.

    This is called on every user prompt to determine if
    RLM mode should activate.
    """
    try:
        from src.complexity_classifier import extract_complexity_signals, should_activate_rlm
        from src.types import SessionContext
        
        # Create minimal context (full context comes from Claude Code)
        context = SessionContext()
        
        should_activate, reason = should_activate_rlm(prompt, context)
        
        result = {
            "activate_rlm": should_activate,
            "reason": reason,
        }
        
        # Output as JSON for hook processing
        print(json.dumps(result))
        
    except ImportError:
        # Fallback if modules not available
        result = {
            "activate_rlm": False,
            "reason": "modules_not_loaded",
        }
        print(json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = sys.stdin.read()
    
    check_complexity(prompt)
