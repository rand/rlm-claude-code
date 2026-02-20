"""
Benchmark test configuration.

Provides an opt-in bounded benchmark mode for restricted or CI environments:
    RLM_BENCHMARK_BOUNDED=1
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@pytest.fixture
def bounded_benchmark(benchmark: Any) -> Callable[..., Any]:
    """
    Benchmark runner with optional deterministic bounds.

    Default mode uses pytest-benchmark behavior unchanged.
    Bounded mode uses benchmark.pedantic with environment-controlled limits:
      - RLM_BENCHMARK_ROUNDS (default: 1)
      - RLM_BENCHMARK_ITERATIONS (default: 1)
      - RLM_BENCHMARK_WARMUP_ROUNDS (default: 0)
    """

    if not _env_enabled("RLM_BENCHMARK_BOUNDED"):
        return benchmark

    rounds = max(1, _env_int("RLM_BENCHMARK_ROUNDS", 1))
    iterations = max(1, _env_int("RLM_BENCHMARK_ITERATIONS", 1))
    warmup_rounds = max(0, _env_int("RLM_BENCHMARK_WARMUP_ROUNDS", 0))

    def run(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return benchmark.pedantic(
            func,
            args=args,
            kwargs=kwargs,
            rounds=rounds,
            iterations=iterations,
            warmup_rounds=warmup_rounds,
        )

    return run
