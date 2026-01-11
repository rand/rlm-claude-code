Bypass RLM mode for a simple operation.

## Usage

`/simple [command]` — Execute command without RLM orchestration

## When to Use

Use `/simple` when you know the task doesn't need RLM's context decomposition:

- Quick file reads: `/simple show package.json`
- Simple commands: `/simple run npm test`
- Git operations: `/simple git status`
- Acknowledgments: `/simple yes, continue`

## How It Works

The `/simple` prefix tells RLM to skip complexity analysis and execute directly through standard Claude Code. This:

1. Avoids the ~50ms complexity classifier overhead
2. Skips context externalization
3. Uses direct model call without REPL
4. Still maintains conversation history

## Examples

```
/simple cat README.md
/simple git diff HEAD~1
/simple npm install lodash
/simple ls -la src/
```

## When NOT to Use

Don't use `/simple` for:
- Multi-file operations
- Debugging with large output
- Refactoring tasks
- Anything that says "find all" or "across"
- Questions about code relationships

These benefit from RLM's context decomposition and recursive analysis.
