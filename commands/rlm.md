Toggle or configure RLM (Recursive Language Model) mode.

## Usage

- `/rlm` — Show current RLM status and configuration
- `/rlm on` — Force RLM mode for this session
- `/rlm off` — Disable RLM mode (use standard Claude Code)
- `/rlm auto` — Use complexity-based activation (default)
- `/rlm verbose` — Enable verbose trajectory output
- `/rlm debug` — Enable debug trajectory output with full content

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
