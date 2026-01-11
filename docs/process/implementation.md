# Implementation Workflow

This document describes the workflow for implementing RLM-Claude-Code features.

## Phase Status

Update this section as phases complete:

| Phase | Status | Start Date | Complete Date |
|-------|--------|------------|---------------|
| 1. Core Infrastructure | Complete | 2026-01-10 | 2026-01-10 |
| 2. Claude Code Integration | Complete | 2026-01-11 | 2026-01-11 |
| 3. Optimization | In Progress | 2026-01-11 | |
| 4. Advanced Features | Not Started | | |

## Implementation Order

### Phase 1: Core Infrastructure

**Goal**: Working RLM loop with context externalization and REPL.

```
1.1 context_manager.py
    ├── SessionContext dataclass
    ├── externalize_conversation()
    ├── externalize_files()
    └── externalize_tool_outputs()

1.2 repl_environment.py
    ├── RLMEnvironment class
    ├── Sandbox setup (RestrictedPython)
    ├── Built-in helpers (peek, search)
    └── Tool integration (pydantic, hypothesis)

1.3 recursive_handler.py
    ├── RecursiveREPL class
    ├── Depth management
    ├── Sub-call routing
    └── Result aggregation

1.4 router_integration.py
    ├── Model selection by depth
    ├── claude-code-router config
    └── Fallback handling
```

**Validation Checkpoint**:
```bash
# NIAH test: Find needle in 500K token haystack
pytest tests/integration/test_niah.py -v
```

### Phase 2: Claude Code Integration

**Goal**: RLM mode activates automatically, tools work normally.

```
2.1 complexity_classifier.py
    ├── TaskComplexitySignals dataclass
    ├── extract_complexity_signals()
    ├── should_activate_rlm()
    └── is_definitely_simple()

2.2 hooks/hooks.json
    ├── SessionStart → init RLM
    ├── UserPromptSubmit → check complexity
    └── PreCompact → externalize context

2.3 Tool preservation
    ├── bash tool works in RLM mode
    ├── edit tool works in RLM mode
    ├── read tool works in RLM mode
    └── State sync after tool use

2.4 State persistence
    ├── Save RLM state on session end
    ├── Restore RLM state on session resume
    └── Clear state on /clear
```

**Validation Checkpoint**:
```bash
# Real coding task: bug fix workflow
pytest tests/integration/test_bug_fix_workflow.py -v
```

### Phase 3: Optimization

**Goal**: RLM is fast enough for interactive use.

```
3.1 Async sub-calls
    ├── Parallel recursive queries
    ├── Request batching
    └── Cancellation support

3.2 Caching
    ├── Summarization cache
    ├── Frequently-accessed context
    └── REPL state persistence

3.3 Cost tracking
    ├── Token counting per component
    ├── Cost estimation before execution
    └── Budget alerts

3.4 Prompt optimization
    ├── Root prompt variants A/B test
    ├── Recursive prompt variants
    └── Strategy-specific prompts
```

**Validation Checkpoint**:
```bash
# Performance benchmarks
pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json
```

### Phase 4: Advanced Features

**Goal**: Production-ready with full observability.

```
4.1 Depth=2 with REPL
    ├── Child REPL creation
    ├── State isolation
    └── Result propagation

4.2 Smart routing
    ├── Query type detection
    ├── Model selection by query
    └── Fallback chains

4.3 Learning
    ├── Strategy success tracking
    ├── Prompt adaptation
    └── User feedback integration

4.4 Visualization
    ├── Trajectory viewer (web)
    ├── Replay capability
    └── Export formats (JSON, HTML)
```

**Validation Checkpoint**:
```bash
# Full integration tests
pytest tests/ -v --cov=src/ --cov-report=html
```

## File Creation Order

When starting fresh, create files in this order to maintain valid imports:

```
src/
├── __init__.py                    # 1. Package init
├── types.py                       # 2. Shared types (no deps)
├── config.py                      # 3. Configuration loading
├── context_manager.py             # 4. Context handling
├── repl_environment.py            # 5. REPL sandbox
├── recursive_handler.py           # 6. Recursive calls
├── complexity_classifier.py       # 7. Activation logic
├── trajectory.py                  # 8. Trajectory events
├── router_integration.py          # 9. Model routing
└── orchestrator.py                # 10. Main loop (imports all)
```

## Development Commands

```bash
# Start new feature
git checkout -b feature/component-name

# Watch mode for types
ty check src/ --watch

# Watch mode for tests
pytest-watch tests/ -v

# Interactive REPL testing
python -m src.repl_environment --interactive

# Generate trajectory for debugging
python -m src.orchestrator --query "test query" --trace

# Profile performance
python -m cProfile -o profile.out -m src.orchestrator
```

## Spec Traceability

Every implementation must reference the spec section it implements:

```python
def should_activate_rlm(prompt: str, context: SessionContext) -> tuple[bool, str]:
    """
    Determine if RLM mode should activate.
    
    Implements: Spec §6.3 Task Complexity-Based Activation
    
    Returns:
        (should_activate, reason) tuple
    """
```

When changing behavior, update spec first:

1. Identify spec section affected
2. Update spec with new behavior
3. Get approval on spec change
4. Implement code matching updated spec
5. Add test verifying spec compliance

## Error Handling Patterns

### Expected Errors (user-facing)

```python
class RLMError(Exception):
    """Base class for RLM errors."""
    pass

class ContextTooLargeError(RLMError):
    """Context exceeds maximum size for RLM processing."""
    def __init__(self, size: int, max_size: int):
        self.size = size
        self.max_size = max_size
        super().__init__(f"Context size {size} exceeds maximum {max_size}")

class RecursionDepthError(RLMError):
    """Maximum recursion depth exceeded."""
    pass
```

### Unexpected Errors (internal)

```python
# Log full context, don't expose to user
try:
    result = await repl.execute(code)
except Exception as e:
    logger.exception("REPL execution failed", extra={
        "code": code[:500],
        "depth": self.depth,
        "context_size": len(self.context)
    })
    # Graceful degradation
    return FallbackResult(reason="repl_error")
```

## Commit Message Format

```
[component] Brief description

- Detailed change 1
- Detailed change 2

Implements: Spec §X.Y
Tests: tests/test_component.py
```

Example:
```
[repl] Add pydantic and hypothesis to sandbox

- Install pydantic>=2.0 and hypothesis>=6.0 in REPL env
- Add to allowed imports in sandbox globals
- Add security tests for subprocess restrictions

Implements: Spec §4.1.1
Tests: tests/test_repl_security.py
```
