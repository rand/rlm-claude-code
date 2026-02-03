"""
Benchmark tests for Phase 3 optimization modules.

Implements: Spec §8.2 Performance Targets validation

Run with:
    pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json
"""

import asyncio
import tempfile
from pathlib import Path

from src.cache import (
    ContextCache,
    LRUCache,
    REPLStateCache,
    SummarizationCache,
)
from src.cost_tracker import CostComponent, CostTracker, estimate_tokens
from src.prompt_optimizer import PromptLibrary, PromptResult, PromptType


class TestCacheBenchmarks:
    """Benchmark tests for caching operations."""

    def test_lru_cache_put_performance(self, benchmark):
        """Benchmark LRU cache put operations."""
        cache: LRUCache[str] = LRUCache(max_entries=1000)

        def do_puts():
            for i in range(100):
                cache.put(f"key_{i}", f"value_{i}" * 100)

        benchmark(do_puts)

    def test_lru_cache_get_performance(self, benchmark):
        """Benchmark LRU cache get operations (hits)."""
        cache: LRUCache[str] = LRUCache(max_entries=1000)
        # Pre-populate
        for i in range(100):
            cache.put(f"key_{i}", f"value_{i}" * 100)

        def do_gets():
            for i in range(100):
                cache.get(f"key_{i}")

        benchmark(do_gets)

    def test_lru_cache_mixed_operations(self, benchmark):
        """Benchmark mixed cache operations."""
        cache: LRUCache[str] = LRUCache(max_entries=500)

        def do_mixed():
            for i in range(50):
                cache.put(f"key_{i}", f"value_{i}" * 100)
                cache.get(f"key_{i}")
                if i % 10 == 0:
                    cache.invalidate(f"key_{i}")

        benchmark(do_mixed)

    def test_summarization_cache_performance(self, benchmark):
        """Benchmark summarization cache operations."""
        cache = SummarizationCache(max_entries=100)
        content = "x" * 10000  # 10KB content

        def do_cache_ops():
            for i in range(20):
                summary = f"Summary {i}"
                cache.put_summary(content + str(i), summary)
                cache.get_summary(content + str(i))

        benchmark(do_cache_ops)

    def test_context_cache_file_operations(self, benchmark):
        """Benchmark context cache file operations."""
        cache = ContextCache(max_files=100)
        file_content = "def foo():\n    pass\n" * 100

        def do_file_ops():
            for i in range(50):
                cache.put_file(f"/path/to/file_{i}.py", file_content)
                cache.get_file(f"/path/to/file_{i}.py")

        benchmark(do_file_ops)

    def test_repl_state_cache_save_load(self, benchmark):
        """Benchmark REPL state save/load operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = REPLStateCache(cache_dir=Path(tmpdir))
            state = {
                "variables": {f"var_{i}": i for i in range(50)},
                "history": [f"command_{i}" for i in range(100)],
            }

            def do_save_load():
                for i in range(10):
                    cache.save_state(f"session_{i}", state)
                    cache.load_state(f"session_{i}")

            benchmark(do_save_load)


class TestCostTrackerBenchmarks:
    """Benchmark tests for cost tracking operations."""

    def test_record_usage_performance(self, benchmark):
        """Benchmark token usage recording."""
        tracker = CostTracker(budget_tokens=1_000_000)

        def do_records():
            for i in range(100):
                tracker.record_usage(
                    input_tokens=1000,
                    output_tokens=500,
                    model="sonnet",
                    component=CostComponent.ROOT_PROMPT,
                )

        benchmark(do_records)
        tracker.reset()

    def test_estimate_cost_performance(self, benchmark):
        """Benchmark cost estimation."""
        tracker = CostTracker()

        def do_estimates():
            for i in range(100):
                tracker.estimate_cost(
                    prompt_length=4000,
                    expected_output_length=2000,
                    model="sonnet",
                    component=CostComponent.RECURSIVE_CALL,
                )

        benchmark(do_estimates)

    def test_breakdown_calculation_performance(self, benchmark):
        """Benchmark breakdown calculations with many records."""
        tracker = CostTracker(budget_tokens=10_000_000)
        # Pre-populate with records
        components = list(CostComponent)
        models = ["opus", "sonnet", "haiku"]
        for i in range(500):
            tracker.record_usage(
                input_tokens=1000 + i,
                output_tokens=500 + i,
                model=models[i % 3],
                component=components[i % len(components)],
            )

        def do_breakdowns():
            tracker.get_breakdown_by_component()
            tracker.get_breakdown_by_model()
            tracker.get_summary()

        benchmark(do_breakdowns)

    def test_token_estimation_performance(self, benchmark):
        """Benchmark token estimation for various text sizes."""
        texts = [
            "short text",
            "medium text " * 100,
            "long text " * 1000,
        ]

        def do_estimates():
            for text in texts:
                for _ in range(10):
                    estimate_tokens(text)

        benchmark(do_estimates)


class TestPromptOptimizerBenchmarks:
    """Benchmark tests for prompt optimization operations."""

    def test_variant_selection_performance(self, benchmark):
        """Benchmark variant selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PromptLibrary(storage_path=Path(tmpdir))

            def do_selections():
                for _ in range(50):
                    library.select_variant(PromptType.ROOT, strategy="epsilon_greedy")
                    library.select_variant(PromptType.RECURSIVE, strategy="random")

            benchmark(do_selections)

    def test_result_recording_performance(self, benchmark):
        """Benchmark result recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PromptLibrary(storage_path=Path(tmpdir))

            def do_records():
                for i in range(50):
                    library.record_result(
                        PromptResult(
                            variant_id="root_v1",
                            success=i % 3 != 0,
                            tokens_used=1000 + i * 10,
                            execution_time_ms=500 + i,
                            user_feedback=1 if i % 2 == 0 else -1,
                        )
                    )

            benchmark(do_records)

    def test_prompt_rendering_performance(self, benchmark):
        """Benchmark prompt template rendering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PromptLibrary(storage_path=Path(tmpdir))
            variant = library.get_variant("root_v1")

            context_summary = "Context with " + "data " * 100
            query = "Analyze this " + "complex " * 50 + "query"

            def do_renders():
                for _ in range(100):
                    library.render_prompt(
                        variant,
                        context_summary=context_summary,
                        query=query,
                    )

            benchmark(do_renders)

    def test_recommendations_performance(self, benchmark):
        """Benchmark recommendation generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PromptLibrary(storage_path=Path(tmpdir))

            # Pre-populate with results
            for i in range(100):
                library.record_result(
                    PromptResult(
                        variant_id="root_v1" if i % 2 == 0 else "root_v2",
                        success=i % 4 != 0,
                        tokens_used=5000 + i * 50,
                        execution_time_ms=1000 + i * 10,
                        user_feedback=-1 if i % 3 == 0 else 1,
                    )
                )

            def do_recommendations():
                for _ in range(10):
                    library.get_recommendations(PromptType.ROOT)

            benchmark(do_recommendations)


class TestAsyncBenchmarks:
    """Benchmark tests for async operations."""

    def test_parallel_task_overhead(self, benchmark):
        """Benchmark overhead of parallel task creation."""

        async def dummy_task(i: int) -> int:
            await asyncio.sleep(0.001)
            return i * 2

        async def run_parallel():
            tasks = [asyncio.create_task(dummy_task(i)) for i in range(10)]
            return await asyncio.gather(*tasks)

        def do_parallel():
            return asyncio.run(run_parallel())

        benchmark(do_parallel)

    def test_semaphore_overhead(self, benchmark):
        """Benchmark semaphore acquisition overhead."""

        async def acquire_release():
            semaphore = asyncio.Semaphore(5)
            for _ in range(20):
                async with semaphore:
                    await asyncio.sleep(0.0001)

        def do_semaphore():
            asyncio.run(acquire_release())

        benchmark(do_semaphore)


class TestIntegrationBenchmarks:
    """Integration benchmarks for complete workflows."""

    def test_full_cache_workflow(self, benchmark):
        """Benchmark complete caching workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_cache = ContextCache()
            summary_cache = SummarizationCache()
            repl_cache = REPLStateCache(cache_dir=Path(tmpdir))

            def do_workflow():
                # Simulate typical RLM workflow
                for i in range(10):
                    # Cache files
                    context_cache.put_file(f"/src/file_{i}.py", "code " * 500)

                    # Cache summaries
                    content = f"Large content block {i} " * 200
                    summary_cache.put_summary(content, f"Summary of block {i}")

                    # Save/load REPL state
                    repl_cache.save_state(f"session_{i}", {"step": i})
                    repl_cache.load_state(f"session_{i}")

                    # Retrieve cached data
                    context_cache.get_file(f"/src/file_{i}.py")
                    summary_cache.get_summary(content)

            benchmark(do_workflow)

    def test_full_cost_tracking_workflow(self, benchmark):
        """Benchmark complete cost tracking workflow."""
        tracker = CostTracker(budget_tokens=500_000, budget_dollars=10.0)

        def do_workflow():
            # Simulate RLM session
            for depth in range(3):
                for call in range(5):
                    # Estimate before call
                    estimate = tracker.estimate_cost(
                        prompt_length=2000 * (depth + 1),
                        expected_output_length=1000,
                        model=["opus", "sonnet", "haiku"][depth],
                        component=CostComponent.RECURSIVE_CALL,
                    )

                    # Check budget
                    tracker.would_exceed_budget(estimate)

                    # Record actual usage
                    tracker.record_usage(
                        input_tokens=estimate.estimated_input_tokens,
                        output_tokens=estimate.estimated_output_tokens,
                        model=estimate.model,
                        component=estimate.component,
                    )

            # Get summary
            tracker.get_summary()

        benchmark(do_workflow)
        tracker.reset()
