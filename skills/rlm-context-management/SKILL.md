# RLM Context Management Skill

## When to Activate

This skill activates when:
- Context exceeds complexity threshold (cross-file reasoning, debugging with large output, multi-module tasks)
- User explicitly requests RLM mode (`/rlm`)
- Task requires verification chains (depth=2 patterns)

## Capabilities

### Context Externalization
- Conversation history → `conversation` variable
- Cached files → `files` dict
- Tool outputs → `tool_outputs` list
- Session state → `working_memory` dict

### REPL Operations
- `peek(var, start, end)` — View portion of context
- `search(var, pattern, regex=False)` — Find patterns in context
- `summarize(var, max_tokens=500)` — Summarize via sub-call
- `recursive_query(query, context)` — Spawn sub-RLM call

### Verification Tools
- `pydantic` — Schema validation for structured data
- `hypothesis` — Property-based testing
- `cpmpy` — Constraint programming (available as `cp`)

## Strategy Selection

### For Large Context (>80K tokens)
1. Peek first 2K chars to understand structure
2. Search for relevant patterns
3. Partition and map over chunks if needed
4. Recursive queries for semantic understanding

### For Debugging Tasks
1. Peek recent tool output for error
2. Search codebase for relevant files
3. Recursive query to analyze each candidate
4. Verify fix won't break other code (depth=2)

### For Multi-File Refactoring
1. Identify all affected files via search
2. Recursive query for each file's current state
3. Plan changes with dependency awareness
4. Verify with CPMpy constraint model (depth=2)

## Output Protocol

Signal completion with:
- `FINAL(answer)` — Direct answer
- `FINAL_VAR(var_name)` — Answer stored in variable

## Trajectory

All RLM operations emit trajectory events for visibility:
- ANALYZE, REPL, REASON, RECURSE, FINAL
- Streaming output with configurable verbosity
- JSON export for debugging

## Configuration

```json
{
  "rlm": {
    "activation": {"mode": "complexity"},
    "depth": {"default": 2, "max": 3},
    "trajectory": {"verbosity": "normal", "streaming": true}
  }
}
```

## References

- Spec: rlm-claude-code-spec.md
- RLM Paper: https://arxiv.org/abs/2512.24601v1
- CPMpy Docs: https://cpmpy.readthedocs.io/
