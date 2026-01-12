Toggle or configure RLM (Recursive Language Model) mode.

## Usage

### Basic Commands
- `/rlm` — Show current RLM status and configuration
- `/rlm on` — Enable RLM auto-activation for this session
- `/rlm off` — Disable RLM mode (use standard Claude Code)

### Execution Mode
- `/rlm mode fast` — Quick responses, shallow analysis, cheaper models
- `/rlm mode balanced` — Standard processing (default)
- `/rlm mode thorough` — Deep analysis, multiple passes, powerful models

### Budget Controls
- `/rlm budget $5` — Set session budget to $5
- `/rlm budget` — Show current budget

### Depth Configuration
- `/rlm depth 2` — Set max recursion depth (0-3)
- `/rlm depth` — Show current depth setting

### Model Preferences
- `/rlm model opus` — Force specific model (opus, sonnet, haiku, gpt-4o, codex)
- `/rlm model auto` — Use automatic model selection

### Tool Access for Sub-LLMs
- `/rlm tools none` — Pure reasoning only
- `/rlm tools repl` — Python REPL only
- `/rlm tools read` — REPL + file reading
- `/rlm tools full` — Full tool access

### Trajectory Verbosity
- `/rlm verbosity minimal` — RECURSE, FINAL, ERROR only
- `/rlm verbosity normal` — All events, truncated (default)
- `/rlm verbosity verbose` — All events, full content
- `/rlm verbosity debug` — Everything + internal state

### Other
- `/rlm status` — Show full configuration
- `/rlm reset` — Reset to defaults
- `/rlm save` — Save preferences to disk

## Execution Modes

| Mode | Depth | Model Tier | Tool Access | Budget |
|------|-------|------------|-------------|--------|
| fast | 1 | Haiku/4o-mini | REPL only | $0.50 |
| balanced | 2 | Sonnet/GPT-4o | Read-only | $2.00 |
| thorough | 3 | Opus/GPT-5 | Full | $10.00 |

## When to Use

**Force RLM on** when:
- Working with large codebases (>50 files)
- Debugging complex multi-file issues
- Refactoring across module boundaries
- You want to see the reasoning trajectory

**Use thorough mode** when:
- Architecture design decisions
- Complex debugging sessions
- Comprehensive code reviews

**Use fast mode** when:
- Quick lookups
- Simple file operations
- Tight time constraints

## Configuration File

Preferences are saved to `~/.config/rlm-claude-code/preferences.json`

## Instructions

When the user runs `/rlm <command>`, use the `handle_rlm_command` function from `src/user_preferences.py` to process the command and display the result.

For example:
- `/rlm mode fast` → Call `handle_rlm_command("mode fast")`
- `/rlm budget $10` → Call `handle_rlm_command("budget $10")`
- `/rlm status` → Call `handle_rlm_command("status")`

The function returns a status message to display to the user.
