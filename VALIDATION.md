# Session Isolation Validation Guide

## Overview

This document describes how to validate session isolation in multi-terminal workflows.

## Prerequisites

1. Install the forked rlm-claude-code plugin
2. Configure Claude Code to use the plugin via hooks

## Test Scenario: Two Terminals on Same Project

### Setup

1. Open Terminal A in a project directory
2. Open Terminal B in the same project directory
3. Both terminals should have Claude Code with rlm-claude-code hooks enabled

### Terminal A Task

1. Start a Claude Code session
2. Work on a specific feature (e.g., authentication)
3. Add facts to memory:
   ```
   memory_add_fact("Auth uses JWT tokens", tier="session")
   memory_add_fact("Current task: implement login flow", tier="task")
   ```

### Terminal B Task

1. Start a separate Claude Code session
2. Work on a different feature (e.g., billing)
3. Add facts to memory:
   ```
   memory_add_fact("Billing uses Stripe API", tier="session")
   memory_add_fact("Current task: payment processing", tier="task")
   ```

### Verification Checks

#### Check 1: Session Tier Isolation
In Terminal A, query session tier:
```python
results = memory_query("JWT", tier="session")
# Should return: Auth uses JWT tokens
# Should NOT contain: Billing uses Stripe API
```

In Terminal B, query session tier:
```python
results = memory_query("Stripe", tier="session")
# Should return: Billing uses Stripe API
# Should NOT contain: Auth uses JWT tokens
```

#### Check 2: Task Tier Isolation
In Terminal A:
```python
results = memory_query("task", tier="task")
# Should only see: implement login flow
# Should NOT see: payment processing
```

#### Check 3: Longterm Tier Sharing
Add a longterm fact in Terminal A:
```python
memory_add_fact("PostgreSQL runs on port 5433", tier="longterm")
```

Query from Terminal B:
```python
results = memory_query("PostgreSQL", tier="longterm")
# Should return: PostgreSQL runs on port 5433
```

### Expected Results

| Check | Terminal A | Terminal B | Result |
|-------|-----------|-----------|--------|
| Session isolation | Sees own session facts | Sees own session facts | PASS if isolated |
| Task isolation | Sees own task facts | Sees own task facts | PASS if isolated |
| Longterm sharing | Sees shared facts | Sees shared facts | PASS if visible to both |

## Automated Test Coverage

The following automated tests validate session isolation:

- `tests/unit/test_session_isolation.py` - Unit tests for query filtering
- `tests/integration/test_parallel_sessions.py` - Concurrent access tests

Run with:
```bash
uv run pytest tests/unit/test_session_isolation.py tests/integration/test_parallel_sessions.py -v
```

## Troubleshooting

### Facts Not Isolated

1. Verify `CLAUDE_SESSION_ID` environment variable is set
2. Check that hooks are properly configured
3. Ensure `metadata.session_id` is being stored with facts

### Longterm Facts Not Visible

1. Longterm tier should NOT filter by session_id
2. Verify tier is correctly set to "longterm"
