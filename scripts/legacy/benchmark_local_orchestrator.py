#!/usr/bin/env python3
"""
Benchmark script for local orchestrator models.

Measures latency, throughput, and decision quality across different
local model configurations.

Usage:
    uv run python scripts/benchmark_local_orchestrator.py
    uv run python scripts/benchmark_local_orchestrator.py --model gemma-3-270m-it
    uv run python scripts/benchmark_local_orchestrator.py --backend ollama
    uv run python scripts/benchmark_local_orchestrator.py --queries 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Import from installed package
from src import (
    LocalOrchestrator,
    LocalModelConfig,
    RECOMMENDED_CONFIGS,
)
from src.local_orchestrator import LocalModelBackend


# Sample queries for benchmarking
BENCHMARK_QUERIES = [
    # Simple queries (should NOT activate RLM)
    ("show config.py", "simple_read"),
    ("yes", "conversational"),
    ("ok thanks", "conversational"),
    ("what is the syntax for list comprehension?", "knowledge"),
    ("fix the typo in README.md", "narrow_scope"),

    # Discovery queries (should activate RLM)
    ("Why is the API returning 500 errors?", "discovery"),
    ("How does authentication work in this codebase?", "discovery"),
    ("Where is the database connection configured?", "discovery"),

    # Synthesis queries (should activate RLM)
    ("Update all usages of the deprecated API", "synthesis"),
    ("Find all instances of the old logging pattern", "synthesis"),
    ("Ensure consistent error handling across services", "synthesis"),

    # Debugging queries (should activate RLM with high depth)
    ("This test is flaky and fails randomly", "debugging_deep"),
    ("I think there's a race condition here", "debugging_deep"),
    ("This used to work but now it doesn't", "debugging_deep"),

    # Architectural queries (should activate RLM)
    ("Design a system for handling real-time events", "architecture"),
    ("Should we use microservices or monolith?", "architecture"),
    ("What's the best approach for adding caching?", "uncertainty"),

    # Mixed/ambiguous queries
    ("Make this faster", "ambiguous"),
    ("Clean up the code", "ambiguous"),
    ("Something feels wrong with this", "ambiguous"),
]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    query: str
    query_type: str
    latency_ms: float
    activate_rlm: bool
    expected_activation: bool
    decision: dict
    error: str | None = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""

    model_name: str
    backend: str
    total_queries: int
    successful_queries: int
    failed_queries: int

    # Latency stats (ms)
    latency_mean: float
    latency_median: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_max: float

    # Decision accuracy
    accuracy: float  # % of correct activation decisions
    false_positives: int  # Activated when shouldn't have
    false_negatives: int  # Didn't activate when should have

    # Per-type latencies
    latency_by_type: dict[str, float] = field(default_factory=dict)


def expected_activation(query_type: str) -> bool:
    """Determine expected RLM activation for a query type."""
    activate_types = {
        "discovery", "synthesis", "debugging_deep",
        "architecture", "uncertainty", "ambiguous"
    }
    return query_type in activate_types


async def benchmark_query(
    orchestrator: LocalOrchestrator,
    query: str,
    query_type: str,
    context_summary: str = "- Context tokens: 10,000",
) -> BenchmarkResult:
    """Benchmark a single query."""
    start = time.perf_counter()
    error = None
    decision = {}
    activate = False

    try:
        decision = await orchestrator.orchestrate(query, context_summary)
        activate = decision.get("activate_rlm", False)
    except Exception as e:
        error = str(e)

    latency_ms = (time.perf_counter() - start) * 1000

    return BenchmarkResult(
        query=query,
        query_type=query_type,
        latency_ms=latency_ms,
        activate_rlm=activate,
        expected_activation=expected_activation(query_type),
        decision=decision,
        error=error,
    )


async def run_benchmark(
    config: LocalModelConfig,
    queries: list[tuple[str, str]],
    warmup_runs: int = 3,
    context_summary: str = "- Context tokens: 10,000\n- Files: main.py, utils.py",
) -> BenchmarkSummary:
    """Run benchmark with given config."""
    orchestrator = LocalOrchestrator(config=config)

    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...", end=" ", flush=True)
    for _ in range(warmup_runs):
        try:
            await orchestrator.orchestrate("warmup query", context_summary)
        except Exception:
            pass  # Ignore warmup errors
    print("done")

    # Run benchmarks
    results: list[BenchmarkResult] = []
    print(f"  Running {len(queries)} queries...", end=" ", flush=True)

    for query, query_type in queries:
        result = await benchmark_query(orchestrator, query, query_type, context_summary)
        results.append(result)

    print("done")

    # Calculate statistics
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    if not successful:
        return BenchmarkSummary(
            model_name=config.model_name,
            backend=config.backend.value,
            total_queries=len(results),
            successful_queries=0,
            failed_queries=len(failed),
            latency_mean=0,
            latency_median=0,
            latency_p95=0,
            latency_p99=0,
            latency_min=0,
            latency_max=0,
            accuracy=0,
            false_positives=0,
            false_negatives=0,
        )

    latencies = [r.latency_ms for r in successful]
    latencies_sorted = sorted(latencies)

    # Accuracy
    correct = sum(1 for r in successful if r.activate_rlm == r.expected_activation)
    false_pos = sum(1 for r in successful if r.activate_rlm and not r.expected_activation)
    false_neg = sum(1 for r in successful if not r.activate_rlm and r.expected_activation)

    # Per-type latencies
    latency_by_type: dict[str, list[float]] = {}
    for r in successful:
        if r.query_type not in latency_by_type:
            latency_by_type[r.query_type] = []
        latency_by_type[r.query_type].append(r.latency_ms)

    avg_by_type = {k: statistics.mean(v) for k, v in latency_by_type.items()}

    return BenchmarkSummary(
        model_name=config.model_name,
        backend=config.backend.value,
        total_queries=len(results),
        successful_queries=len(successful),
        failed_queries=len(failed),
        latency_mean=statistics.mean(latencies),
        latency_median=statistics.median(latencies),
        latency_p95=latencies_sorted[int(len(latencies_sorted) * 0.95)] if len(latencies_sorted) > 20 else max(latencies),
        latency_p99=latencies_sorted[int(len(latencies_sorted) * 0.99)] if len(latencies_sorted) > 100 else max(latencies),
        latency_min=min(latencies),
        latency_max=max(latencies),
        accuracy=correct / len(successful) * 100,
        false_positives=false_pos,
        false_negatives=false_neg,
        latency_by_type=avg_by_type,
    )


def print_summary(summary: BenchmarkSummary) -> None:
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print(f"Model: {summary.model_name} ({summary.backend})")
    print(f"{'='*60}")

    print(f"\nQueries: {summary.successful_queries}/{summary.total_queries} successful")
    if summary.failed_queries > 0:
        print(f"  (!) {summary.failed_queries} failed")

    print(f"\nLatency (ms):")
    print(f"  Mean:   {summary.latency_mean:>8.2f}")
    print(f"  Median: {summary.latency_median:>8.2f}")
    print(f"  P95:    {summary.latency_p95:>8.2f}")
    print(f"  P99:    {summary.latency_p99:>8.2f}")
    print(f"  Min:    {summary.latency_min:>8.2f}")
    print(f"  Max:    {summary.latency_max:>8.2f}")

    print(f"\nDecision Accuracy: {summary.accuracy:.1f}%")
    print(f"  False positives: {summary.false_positives}")
    print(f"  False negatives: {summary.false_negatives}")

    if summary.latency_by_type:
        print(f"\nLatency by query type:")
        for qtype, latency in sorted(summary.latency_by_type.items()):
            print(f"  {qtype:20s}: {latency:>8.2f} ms")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark local orchestrator models")
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name to benchmark (default: all presets)",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["mlx", "ollama"],
        default=None,
        help="Backend to use",
    )
    parser.add_argument(
        "--preset", "-p",
        choices=list(RECOMMENDED_CONFIGS.keys()),
        default=None,
        help="Use a preset configuration",
    )
    parser.add_argument(
        "--queries", "-n",
        type=int,
        default=len(BENCHMARK_QUERIES),
        help="Number of queries to run (default: all)",
    )
    parser.add_argument(
        "--repeat", "-r",
        type=int,
        default=1,
        help="Repeat each query N times",
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=3,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--heuristics-only",
        action="store_true",
        help="Benchmark heuristics only (no model inference)",
    )

    args = parser.parse_args()

    # Build query list
    queries = BENCHMARK_QUERIES[:args.queries] * args.repeat

    print(f"Benchmarking with {len(queries)} queries")
    print(f"Warmup runs: {args.warmup}")

    summaries: list[BenchmarkSummary] = []

    if args.heuristics_only:
        # Benchmark heuristics only
        print("\n--- Heuristics Only ---")
        config = LocalModelConfig(
            model_name="heuristics",
            fallback_to_heuristics=True,
        )
        # Force heuristics by making backend unavailable
        orchestrator = LocalOrchestrator(config=config)
        orchestrator._get_runner = lambda: (_ for _ in ()).throw(RuntimeError("Forced heuristics"))

        summary = await run_benchmark(config, queries, args.warmup)
        summaries.append(summary)
        print_summary(summary)

    elif args.preset:
        # Single preset
        print(f"\n--- Preset: {args.preset} ---")
        config = RECOMMENDED_CONFIGS[args.preset]
        summary = await run_benchmark(config, queries, args.warmup)
        summaries.append(summary)
        print_summary(summary)

    elif args.model:
        # Single model
        backend = LocalModelBackend.MLX
        if args.backend == "ollama":
            backend = LocalModelBackend.OLLAMA

        print(f"\n--- Model: {args.model} ({backend.value}) ---")
        config = LocalModelConfig(
            model_name=args.model,
            backend=backend,
        )
        summary = await run_benchmark(config, queries, args.warmup)
        summaries.append(summary)
        print_summary(summary)

    else:
        # All presets
        for preset_name, config in RECOMMENDED_CONFIGS.items():
            print(f"\n--- Preset: {preset_name} ---")
            try:
                summary = await run_benchmark(config, queries, args.warmup)
                summaries.append(summary)
                print_summary(summary)
            except Exception as e:
                print(f"  ERROR: {e}")

    # Output JSON if requested
    if args.output and summaries:
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(queries),
            "warmup_runs": args.warmup,
            "summaries": [
                {
                    "model_name": s.model_name,
                    "backend": s.backend,
                    "total_queries": s.total_queries,
                    "successful_queries": s.successful_queries,
                    "latency_mean_ms": s.latency_mean,
                    "latency_median_ms": s.latency_median,
                    "latency_p95_ms": s.latency_p95,
                    "latency_p99_ms": s.latency_p99,
                    "accuracy_pct": s.accuracy,
                    "false_positives": s.false_positives,
                    "false_negatives": s.false_negatives,
                }
                for s in summaries
            ],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\nResults written to {args.output}")

    # Summary comparison
    if len(summaries) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'Backend':<10} {'Mean(ms)':<10} {'P95(ms)':<10} {'Accuracy':<10}")
        print("-" * 65)
        for s in summaries:
            print(f"{s.model_name:<25} {s.backend:<10} {s.latency_mean:<10.2f} {s.latency_p95:<10.2f} {s.accuracy:<10.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
