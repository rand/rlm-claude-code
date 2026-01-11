Run the test suite for RLM-Claude-Code.

## Commands

```bash
# All tests
uv run pytest tests/ -v

# By category
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run pytest tests/property/ -v -m hypothesis
uv run pytest tests/security/ -v
uv run pytest tests/benchmarks/ --benchmark-only

# With coverage
uv run pytest tests/ -v --cov=src/ --cov-report=html

# Specific file
uv run pytest tests/unit/test_complexity_classifier.py -v

# Watch mode
uv run pytest-watch tests/ -v
```

## Coverage Requirements

| Component | Target |
|-----------|--------|
| context_manager.py | 90% |
| repl_environment.py | 85% |
| recursive_handler.py | 90% |
| complexity_classifier.py | 95% |
| trajectory_renderer.py | 80% |

## Quick Checks

Before committing:
1. `uv run pytest tests/unit/ -v` — Unit tests pass
2. `uv run pytest tests/security/ -v` — Security tests pass
3. `uv run ty check src/` — Type check passes
