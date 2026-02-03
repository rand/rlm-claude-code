# ADR-002: Fail-Forward Behavior for Hooks

## Status

Accepted

## Context

RLM and Disciplined Process (DP) hooks can encounter situations where they might want to block or halt execution - test failures, missing specs, validation errors, etc. The observed behavior was that these conditions caused the model to stop working and wait, rather than finding ways to make progress.

User feedback: "When RLM or Disciplined Process would deny or block, the behavior should be to 'fail forward' and find ways to progress, whether it's engaging the user in decision making, trying new things, making suggestions, etc. Just halting is seldom the appropriate move."

## Decision

Adopt a **fail-forward** design principle for all hooks:

1. **Hooks should guide, not gate** - Inject context and suggestions, don't block execution
2. **Errors should be informative** - When something fails, provide actionable information
3. **Offer alternatives** - If the ideal path is blocked, suggest alternatives
4. **Engage the user** - When uncertain, ask rather than halt
5. **Progress over perfection** - Making incremental progress is better than waiting for perfect conditions

### Implementation Guidelines

| Situation | Wrong Approach | Right Approach |
|-----------|---------------|----------------|
| Tests fail | Block commit, halt work | Report failures, suggest fixes, ask if user wants to proceed |
| No spec exists | Refuse to implement | Offer to create minimal spec, or proceed with note |
| Complexity detected | Stop and analyze | Suggest RLM mode, continue working |
| Hook times out | Fail the operation | Log warning, continue with defaults |
| Validation error | Block the action | Report issue, offer workarounds |

### Hook Design Rules

1. **No blocking prompts after tool use** - Causes halting after every action (per ADR-001)
2. **Command hooks should be non-blocking by default** - Set `"blocking": false`
3. **Prompt hooks should offer choices, not mandates** - "Consider..." not "You must..."
4. **Timeouts should fail open** - If a hook times out, allow the operation to proceed
5. **Errors should include suggested next steps** - Don't just report problems, suggest solutions

## Consequences

### Positive
- More fluid, productive interactions
- Users maintain agency over their workflow
- Better handling of edge cases and failures
- Hooks become helpful guidance rather than obstacles

### Negative
- Less enforcement of process discipline
- Users might skip important steps
- Mitigation: Provide clear warnings and end-of-session summaries

### Neutral
- Shifts responsibility for process compliance to user judgment
- Hooks become advisory rather than mandatory

## Examples

### Before (Blocking)
```
Stop hook error: Test suite failed with exit code 144. Must investigate and fix failures before continuing.
```

### After (Fail-Forward)
```
Test suite had failures (exit 144). Options:
1. Run specific failing tests to investigate
2. Continue with current work, address tests later
3. Show test failure details

What would you like to do?
```

## References

- ADR-001: Remove PostToolUse Prompt Hook
- User feedback on halting behavior
