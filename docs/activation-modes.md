# RLM Activation Modes

This guide explains the different activation modes for RLM and how to configure them.

## Overview

RLM can be configured to activate in different ways depending on your workflow preferences. The activation mode controls when RLM automatically engages to help with complex tasks.

## Activation Modes

| Mode | Config Value | When RLM Activates |
|------|--------------|-------------------|
| **Complexity** | `"complexity"` or `"auto"` | When complexity signals are detected (default) |
| **Always** | `"always"` | On every non-trivial prompt |
| **Manual** | `"manual"` or `"never"` | Never auto-activates; requires explicit `/rlm activate` |

## Configuration

Set the activation mode in `~/.claude/rlm-config.json`:

```json
{
  "version": "2.0",
  "activation": {
    "mode": "complexity",
    "fallback_token_threshold": 80000,
    "auto_cross_file": true,
    "dp_phase_aware": true
  }
}
```

## Mode Details

### Complexity Mode (Default)

RLM analyzes each prompt for complexity signals and activates when warranted.

**Complexity signals that trigger activation:**
- Cross-file references (e.g., "How does auth.py interact with api.py?")
- Debugging keywords (e.g., "Why does this fail?", "trace the error")
- Architecture questions (e.g., "How should I structure this?")
- Exhaustive search requests (e.g., "Find all usages of...")
- Security review tasks
- Large context (>80K tokens)

**Best for:** Most users who want RLM to help when needed without manual intervention.

### Always Mode

RLM activates on every prompt except trivial ones (e.g., "ok", "yes", "thanks").

**Best for:**
- Complex coding sessions with frequent multi-step operations
- Users who always want RLM's reasoning capabilities
- Sessions focused on large codebase analysis

**Note:** This mode may add overhead to simple queries, but the fast-path optimization skips trivial prompts automatically.

### Manual Mode

RLM never auto-activates. You must explicitly invoke it with `/rlm-claude-code:rlm activate`.

**Best for:**
- Users who prefer full manual control
- Quick sessions where RLM is rarely needed
- Testing and debugging RLM behavior

## Manual Activation

Regardless of the configured mode, you can always manually trigger RLM:

```
/rlm-claude-code:rlm activate   # Launch orchestrator immediately
/rlm-claude-code:rlm now        # Alias for activate
```

This invokes the RLM orchestrator agent which provides:
- Context decomposition for large inputs
- Recursive sub-queries for complex reasoning
- Memory persistence across sessions
- Intelligent model and depth selection

## Session Control

Control RLM behavior for the current session:

| Command | Effect |
|---------|--------|
| `/rlm-claude-code:rlm on` | Enable RLM for this session (sets mode to always) |
| `/rlm-claude-code:rlm off` | Disable auto-activation for this session |
| `/rlm-claude-code:simple` | Bypass RLM for one query only |

## Environment Variable Override

Disable RLM entirely via environment variable:

```bash
export RLM_DISABLED=1
claude
```

This bypasses all RLM hooks regardless of configuration.

## Fast-Path Optimization

RLM includes a fast-path bypass for trivial prompts that should never trigger activation:

- Simple acknowledgments: "ok", "yes", "no", "thanks"
- Git status commands: "git status", "git log"
- Simple file operations: "read file.py"
- Test commands: "run pytest"

These prompts bypass all complexity checking regardless of activation mode.

## Troubleshooting

### RLM activating too often

Switch to complexity mode or manual mode:
```json
{ "activation": { "mode": "complexity" } }
```

### RLM not activating when expected

1. Check your mode: `cat ~/.claude/rlm-config.json | grep mode`
2. Try manual activation: `/rlm-claude-code:rlm activate`
3. Enable debug logging: `export RLM_DEBUG=1`

### Reset to defaults

```bash
rm ~/.claude/rlm-config.json
# Restart Claude Code to regenerate default config
```

## Related Documentation

- [User Guide](./user-guide.md) - Complete usage documentation
- [Getting Started](./getting-started.md) - Installation and first steps
- [README](../README.md) - Project overview
