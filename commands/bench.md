Run performance benchmarks for RLM-Claude-Code.

## Commands

```bash
# All benchmarks
uv run pytest tests/benchmarks/ --benchmark-only

# With comparison to baseline
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare

# Save results
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json

# Specific benchmark
uv run pytest tests/benchmarks/test_performance.py::test_complexity_classifier_latency --benchmark-only
```

## Performance Targets

| Component | Target | Measurement |
|-----------|--------|-------------|
| Complexity classifier | <50ms | `test_complexity_classifier_latency` |
| REPL execution | <100ms | `test_repl_execution_latency` |
| Trajectory render | <10ms/event | `test_trajectory_render_latency` |
| Context externalization | <500ms | `test_context_externalization` |

## Profiling

```bash
# CPU profiling
python -m cProfile -s cumulative -m src.orchestrator --query "test" 2>&1 | head -50

# Memory profiling
python -m memory_profiler src/orchestrator.py

# Line profiling
kernprof -l -v src/complexity_classifier.py
```

## Cost Tracking

Track token usage:
```bash
uv run python -m src.tools.cost_tracker --trajectory /path/to/trajectory.json
```

Expected costs:
- Simple query through RLM: ~$0.10-0.20
- Complex multi-file task: ~$0.30-0.50
- Deep verification (depth=2): ~$0.50-1.00
