Toggle or configure RLM (Recursive Language Model) mode.

## Usage

- `/rlm-claude-code:rlm` — Show current RLM status and configuration
- `/rlm-claude-code:rlm activate` — Launch RLM orchestrator immediately
- `/rlm-claude-code:rlm now` — Alias for activate (launch orchestrator now)
- `/rlm-claude-code:rlm on` — Enable auto-activation for this session (mode: always)
- `/rlm-claude-code:rlm off` — Disable RLM mode (use standard Claude Code)
- `/rlm-claude-code:rlm auto` — Use complexity-based activation (default)
- `/rlm-claude-code:rlm verbose` — Enable verbose trajectory output
- `/rlm-claude-code:rlm debug` — Enable debug trajectory output with full content

> **Tip**: You can create a shell alias: `alias rlm='claude skill rlm-claude-code:rlm'`

## Activation Commands

| Command | Action |
|---------|--------|
| `activate` or `now` | Immediately launch the RLM orchestrator agent |
| `on` | Enable auto-activation (RLM activates on complex prompts) |
| `off` | Disable auto-activation (manual mode) |
| `auto` | Use complexity-based activation (default) |

When you run `/rlm-claude-code:rlm activate`, the orchestrator agent will be invoked to handle your task with:
- Context decomposition for large inputs
- Recursive sub-queries for complex reasoning
- Memory persistence across sessions
- Intelligent model and depth selection

## Current Configuration

Check `~/.claude/rlm-config.json` for:
- `activation.mode`: "complexity" | "always" | "manual"
- `depth.default`: 2
- `trajectory.verbosity`: "minimal" | "normal" | "verbose" | "debug"

## When to Use

Force RLM on when:
- Working with large codebases (>50 files in context)
- Debugging complex multi-file issues
- Refactoring across module boundaries
- You want to see the reasoning trajectory

Force RLM off when:
- Simple file operations
- Quick questions about single files
- You want faster responses for simple tasks

## Trajectory Verbosity

| Level | Shows |
|-------|-------|
| minimal | RECURSE, FINAL, ERROR only |
| normal | All events, truncated content |
| verbose | All events, full content |
| debug | Everything + internal state |

## Related Commands

- `/rlm-claude-code:rlm-orchestrator` — Launch RLM orchestrator agent for complex context tasks
- `/rlm-claude-code:simple` — Bypass RLM for a single operation
- `/rlm-claude-code:trajectory <file>` — Analyze a saved trajectory
