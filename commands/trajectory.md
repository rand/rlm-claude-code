Analyze an RLM trajectory file.

## Usage

`/trajectory [path]` — Analyze trajectory at path (default: last trajectory)

## Analysis Output

```
Trajectory Analysis: /path/to/trajectory.json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Summary:
  Total events: 23
  Max depth reached: 2
  Recursive calls: 4
  REPL executions: 8
  Tool uses: 3
  Duration: 34.2s
  Estimated cost: $0.47

Event Distribution:
  ANALYZE: 3
  REPL_EXEC: 8
  REASON: 6
  RECURSE_START: 4
  RECURSE_END: 4
  TOOL_USE: 3
  FINAL: 1

Slow Events (>5s):
  [12] RECURSE depth=1: 8.3s - "Analyze auth error handling..."
  [18] TOOL_USE: 5.1s - bash: npm test

Errors: None
```

## Commands

```bash
# Pretty-print trajectory
uv run python -m src.tools.trajectory_viewer trajectory.json

# Filter by depth
uv run python -m src.tools.trajectory_viewer trajectory.json --depth 1

# Filter by type
uv run python -m src.tools.trajectory_viewer trajectory.json --type repl_exec

# Compare two trajectories
uv run python -m src.tools.trajectory_diff working.json failing.json

# Export to HTML
uv run python -m src.tools.trajectory_viewer trajectory.json --html > trajectory.html
```

## Debugging with Trajectories

1. **Capture**: Run with `--verbosity debug --export-trajectory /tmp/debug.json`
2. **Analyze**: `/trajectory /tmp/debug.json`
3. **Identify**: Find slow events, errors, or unexpected recursion
4. **Fix**: See docs/process/debugging.md for issue-specific guidance
