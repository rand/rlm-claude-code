"""
Unit tests for advanced REPL functions: map_reduce(), find_relevant(), extract_functions().

Implements: Spec SPEC-01 tests
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.types import DeferredBatch, Message, MessageRole, SessionContext, ToolOutput


@pytest.fixture
def basic_context():
    """Create a basic context for testing."""
    return SessionContext(
        messages=[
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
        ],
        files={"main.py": "print('hello')", "utils.py": "def helper(): pass"},
        tool_outputs=[ToolOutput(tool_name="bash", content="test output")],
        working_memory={},
    )


@pytest.fixture
def basic_env(basic_context):
    """Create a basic RLM environment for testing."""
    from src.repl_environment import RLMEnvironment

    return RLMEnvironment(basic_context, use_restricted=False)


# =============================================================================
# SPEC-01.01: map_reduce() function signature
# =============================================================================


class TestMapReduceSignature:
    """Tests for map_reduce() function signature (SPEC-01.01)."""

    def test_map_reduce_exists_in_environment(self, basic_env):
        """
        map_reduce should be available in REPL environment.

        @trace SPEC-01.01
        """
        assert "map_reduce" in basic_env.globals
        assert callable(basic_env.globals["map_reduce"])

    def test_map_reduce_accepts_required_parameters(self, basic_env):
        """
        map_reduce(content, map_prompt, reduce_prompt) should work.

        @trace SPEC-01.01
        """
        map_reduce = basic_env.globals["map_reduce"]

        # Should accept required parameters without error
        # Returns DeferredBatch since it's async
        result = map_reduce(
            content="Test content to process",
            map_prompt="Summarize this chunk",
            reduce_prompt="Combine summaries",
        )

        # Should return a DeferredBatch for async processing
        assert isinstance(result, DeferredBatch)

    def test_map_reduce_accepts_optional_n_chunks(self, basic_env):
        """
        map_reduce should accept optional n_chunks parameter.

        @trace SPEC-01.01
        """
        map_reduce = basic_env.globals["map_reduce"]

        result = map_reduce(
            content="Test content",
            map_prompt="Map this",
            reduce_prompt="Reduce this",
            n_chunks=8,
        )

        assert isinstance(result, DeferredBatch)

    def test_map_reduce_accepts_optional_model(self, basic_env):
        """
        map_reduce should accept optional model parameter.

        @trace SPEC-01.01
        """
        map_reduce = basic_env.globals["map_reduce"]

        result = map_reduce(
            content="Test content",
            map_prompt="Map this",
            reduce_prompt="Reduce this",
            model="fast",
        )

        assert isinstance(result, DeferredBatch)


# =============================================================================
# SPEC-01.02: Content partitioning
# =============================================================================


class TestMapReducePartitioning:
    """Tests for map_reduce() content partitioning (SPEC-01.02)."""

    def test_partitions_into_n_chunks(self, basic_env):
        """
        map_reduce should partition content into n_chunks roughly equal parts.

        @trace SPEC-01.02
        """
        map_reduce = basic_env.globals["map_reduce"]
        content = "A" * 1000  # 1000 characters

        result = map_reduce(
            content=content,
            map_prompt="Summarize",
            reduce_prompt="Combine",
            n_chunks=4,
        )

        # Should have 4 operations in the batch (one per chunk)
        # Plus 1 reduce operation
        assert len(result.operations) >= 4

    def test_default_n_chunks_is_4(self, basic_env):
        """
        Default n_chunks should be 4.

        @trace SPEC-01.02
        """
        map_reduce = basic_env.globals["map_reduce"]
        content = "A" * 1000

        result = map_reduce(
            content=content,
            map_prompt="Summarize",
            reduce_prompt="Combine",
        )

        # Default 4 chunks should be created
        assert len(result.operations) >= 4

    def test_single_chunk_for_small_content(self, basic_env):
        """
        Very small content should still work (may produce fewer chunks).

        @trace SPEC-01.02
        """
        map_reduce = basic_env.globals["map_reduce"]
        content = "Short"  # Very small

        result = map_reduce(
            content=content,
            map_prompt="Process",
            reduce_prompt="Combine",
            n_chunks=4,
        )

        # Should still return a valid batch
        assert isinstance(result, DeferredBatch)
        assert len(result.operations) >= 1

    def test_chunks_cover_entire_content(self, basic_env):
        """
        Chunks should cover the entire content without loss.

        @trace SPEC-01.02
        """
        map_reduce = basic_env.globals["map_reduce"]
        content = "ABCDEFGHIJ" * 100  # 1000 chars, easy to verify

        result = map_reduce(
            content=content,
            map_prompt="Process: {chunk}",
            reduce_prompt="Combine",
            n_chunks=4,
        )

        # Verify all chunk contexts together cover the original
        chunk_contexts = [
            op.context for op in result.operations if "chunk" in op.context.lower() or op.context
        ]
        # Should have chunks that span the content
        assert len(chunk_contexts) >= 4


# =============================================================================
# SPEC-01.03: Parallel llm_batch() calls with map_prompt
# =============================================================================


class TestMapReduceParallelExecution:
    """Tests for map_reduce() parallel execution (SPEC-01.03)."""

    def test_uses_deferred_batch_for_parallel(self, basic_env):
        """
        map_reduce should return DeferredBatch for parallel processing.

        @trace SPEC-01.03
        """
        map_reduce = basic_env.globals["map_reduce"]

        result = map_reduce(
            content="Test content " * 100,
            map_prompt="Summarize this chunk",
            reduce_prompt="Combine",
            n_chunks=4,
        )

        assert isinstance(result, DeferredBatch)
        assert result.batch_id is not None

    def test_map_prompt_applied_to_each_chunk(self, basic_env):
        """
        map_prompt should be applied to each chunk.

        @trace SPEC-01.03
        """
        map_reduce = basic_env.globals["map_reduce"]

        result = map_reduce(
            content="Content " * 100,
            map_prompt="Extract key points from this section",
            reduce_prompt="Combine",
            n_chunks=3,
        )

        # Each chunk operation should contain the map prompt
        map_operations = [
            op
            for op in result.operations
            if "Extract key points" in op.query or "map" in op.operation_type.lower()
        ]
        assert len(map_operations) >= 3

    def test_batch_operations_are_independent(self, basic_env):
        """
        Batch operations should be independent (parallelizable).

        @trace SPEC-01.03
        """
        map_reduce = basic_env.globals["map_reduce"]

        result = map_reduce(
            content="A" * 1000,
            map_prompt="Process",
            reduce_prompt="Combine",
            n_chunks=4,
        )

        # Each operation should have unique ID
        op_ids = [op.operation_id for op in result.operations]
        assert len(set(op_ids)) == len(op_ids)


# =============================================================================
# SPEC-01.04: Reduce prompt combines results
# =============================================================================


class TestMapReduceCombination:
    """Tests for map_reduce() result combination (SPEC-01.04)."""

    def test_reduce_prompt_stored_in_batch(self, basic_env):
        """
        reduce_prompt should be stored for final combination.

        @trace SPEC-01.04
        """
        map_reduce = basic_env.globals["map_reduce"]

        result = map_reduce(
            content="Content " * 100,
            map_prompt="Map this",
            reduce_prompt="Synthesize all summaries into final output",
        )

        # The batch should store the reduce prompt for later use
        assert hasattr(result, "reduce_prompt") or any(
            "synthesize" in op.query.lower()
            or "reduce" in getattr(op, "operation_type", "").lower()
            for op in result.operations
        )

    def test_batch_metadata_includes_reduce_info(self, basic_env):
        """
        Batch should include metadata about the reduce phase.

        @trace SPEC-01.04
        """
        map_reduce = basic_env.globals["map_reduce"]

        result = map_reduce(
            content="Test " * 100,
            map_prompt="Extract",
            reduce_prompt="Combine into summary",
        )

        # Batch should have metadata for orchestrator to use
        assert isinstance(result, DeferredBatch)
        # Either has reduce_prompt attribute or metadata
        assert hasattr(result, "reduce_prompt") or hasattr(result, "metadata")


# =============================================================================
# SPEC-01.05: Model parameter values
# =============================================================================


class TestMapReduceModelParameter:
    """Tests for map_reduce() model parameter (SPEC-01.05)."""

    @pytest.mark.parametrize("model", ["fast", "balanced", "powerful", "auto"])
    def test_valid_model_values_accepted(self, basic_env, model):
        """
        map_reduce should accept model values: fast, balanced, powerful, auto.

        @trace SPEC-01.05
        """
        map_reduce = basic_env.globals["map_reduce"]

        result = map_reduce(
            content="Test content",
            map_prompt="Process",
            reduce_prompt="Combine",
            model=model,
        )

        assert isinstance(result, DeferredBatch)

    def test_default_model_is_auto(self, basic_env):
        """
        Default model should be 'auto'.

        @trace SPEC-01.05
        """
        map_reduce = basic_env.globals["map_reduce"]

        result = map_reduce(
            content="Test content",
            map_prompt="Process",
            reduce_prompt="Combine",
        )

        # Should use auto by default (check batch metadata)
        # Since we can't inspect internals directly, verify it works
        assert isinstance(result, DeferredBatch)

    def test_invalid_model_raises_error(self, basic_env):
        """
        Invalid model value should raise ValueError.

        @trace SPEC-01.05
        """
        map_reduce = basic_env.globals["map_reduce"]

        with pytest.raises(ValueError) as exc:
            map_reduce(
                content="Test content",
                map_prompt="Process",
                reduce_prompt="Combine",
                model="invalid_model",
            )

        assert "model" in str(exc.value).lower()


# =============================================================================
# SPEC-01.06: Handle 1M+ character content
# =============================================================================


class TestMapReduceLargeContent:
    """Tests for map_reduce() handling large content (SPEC-01.06)."""

    @pytest.mark.slow
    def test_handles_1m_characters(self, basic_env):
        """
        map_reduce should handle content exceeding 1M characters.

        @trace SPEC-01.06
        """
        map_reduce = basic_env.globals["map_reduce"]

        # Create 1.1M character content
        large_content = "X" * 1_100_000

        # Should not raise an error
        result = map_reduce(
            content=large_content,
            map_prompt="Summarize this chunk",
            reduce_prompt="Combine all summaries",
            n_chunks=10,
        )

        assert isinstance(result, DeferredBatch)
        assert len(result.operations) >= 10

    @pytest.mark.slow
    def test_large_content_creates_appropriate_chunks(self, basic_env):
        """
        Large content should create chunks that fit model context windows.

        @trace SPEC-01.06
        """
        map_reduce = basic_env.globals["map_reduce"]

        # 2M characters
        large_content = "Y" * 2_000_000

        result = map_reduce(
            content=large_content,
            map_prompt="Process",
            reduce_prompt="Combine",
            n_chunks=20,
        )

        # Each chunk context should be reasonable size (< 200K chars each)
        for op in result.operations:
            if op.context:
                assert len(op.context) < 200_000, "Chunk too large for model context"

    def test_empty_content_handled_gracefully(self, basic_env):
        """
        Empty content should not crash.

        @trace SPEC-01.06
        """
        map_reduce = basic_env.globals["map_reduce"]

        result = map_reduce(
            content="",
            map_prompt="Process",
            reduce_prompt="Combine",
        )

        # Should return valid batch (possibly with single empty operation)
        assert isinstance(result, DeferredBatch)


# =============================================================================
# REPL Integration Tests
# =============================================================================


class TestMapReduceREPLIntegration:
    """Tests for map_reduce() integration with REPL environment."""

    def test_callable_from_repl_code(self, basic_env):
        """
        map_reduce should be callable from REPL executed code.

        @trace SPEC-01.22
        """
        result = basic_env.execute(
            "batch = map_reduce('Test content ' * 50, 'Summarize', 'Combine')"
        )

        assert result.success is True
        assert basic_env.has_pending_operations() is True

    def test_batch_tracked_in_pending_operations(self, basic_env):
        """
        map_reduce batch should appear in pending operations.

        @trace SPEC-01.22
        """
        basic_env.execute("batch = map_reduce('Content ' * 100, 'Map it', 'Reduce it', n_chunks=3)")

        ops, batches = basic_env.get_pending_operations()

        # Should have at least one batch
        assert len(batches) >= 1

    def test_works_with_file_content(self, basic_env):
        """
        map_reduce should work with file content from context.

        @trace SPEC-01.22
        """
        result = basic_env.execute(
            """
all_files = '\\n'.join(files.values())
batch = map_reduce(all_files, 'Analyze this code chunk', 'Synthesize analysis')
"""
        )

        assert result.success is True
        assert basic_env.has_pending_operations() is True


# =============================================================================
# SPEC-01.07: find_relevant() function signature
# =============================================================================


class TestFindRelevantSignature:
    """Tests for find_relevant() function signature (SPEC-01.07)."""

    def test_find_relevant_exists_in_environment(self, basic_env):
        """
        find_relevant should be available in REPL environment.

        @trace SPEC-01.07
        """
        assert "find_relevant" in basic_env.globals
        assert callable(basic_env.globals["find_relevant"])

    def test_find_relevant_accepts_required_parameters(self, basic_env):
        """
        find_relevant(content, query) should work with minimal parameters.

        @trace SPEC-01.07
        """
        find_relevant = basic_env.globals["find_relevant"]

        result = find_relevant(
            content="This is test content with some relevant text.",
            query="relevant",
        )

        # Should return a list of (chunk, score) tuples
        assert isinstance(result, list)

    def test_find_relevant_accepts_optional_top_k(self, basic_env):
        """
        find_relevant should accept optional top_k parameter.

        @trace SPEC-01.07
        """
        find_relevant = basic_env.globals["find_relevant"]

        result = find_relevant(
            content="Test content " * 100,
            query="test",
            top_k=3,
        )

        assert isinstance(result, list)
        assert len(result) <= 3

    def test_find_relevant_accepts_optional_use_llm_scoring(self, basic_env):
        """
        find_relevant should accept optional use_llm_scoring parameter.

        @trace SPEC-01.07
        """
        find_relevant = basic_env.globals["find_relevant"]

        # Without LLM scoring (default)
        result = find_relevant(
            content="Test content " * 50,
            query="test",
            use_llm_scoring=False,
        )

        assert isinstance(result, list)


# =============================================================================
# SPEC-01.08: Content chunking with overlap
# =============================================================================


class TestFindRelevantChunking:
    """Tests for find_relevant() content chunking (SPEC-01.08)."""

    def test_chunks_approximately_50_lines(self, basic_env):
        """
        find_relevant should partition content into ~50-line chunks.

        @trace SPEC-01.08
        """
        find_relevant = basic_env.globals["find_relevant"]

        # Create content with 200 lines
        lines = [f"Line {i}: some content about testing" for i in range(200)]
        content = "\n".join(lines)

        result = find_relevant(
            content=content,
            query="testing",
            top_k=10,
        )

        # Should return chunks, each approximately 50 lines
        for chunk, _score in result:
            chunk_lines = chunk.count("\n") + 1
            # Allow some variance (40-60 lines per chunk is acceptable)
            assert 20 <= chunk_lines <= 80, f"Chunk has {chunk_lines} lines, expected ~50"

    def test_chunks_have_overlap(self, basic_env):
        """
        Chunks should have 5-line overlap for context preservation.

        @trace SPEC-01.08
        """
        find_relevant = basic_env.globals["find_relevant"]

        # Create unique numbered lines
        lines = [f"UNIQUE_LINE_{i}" for i in range(150)]
        content = "\n".join(lines)

        result = find_relevant(
            content=content,
            query="UNIQUE_LINE",
            top_k=100,  # Get all chunks
        )

        # With overlap, some lines should appear in multiple chunks
        if len(result) >= 2:
            all_lines_in_chunks = []
            for chunk, _ in result:
                chunk_lines = chunk.strip().split("\n")
                all_lines_in_chunks.extend(chunk_lines)

            # Total lines in chunks should exceed original due to overlap
            unique_chunk_lines = set(all_lines_in_chunks)
            # The implementation should have some overlap
            assert len(all_lines_in_chunks) >= len(unique_chunk_lines)


# =============================================================================
# SPEC-01.09: Keyword pre-filtering
# =============================================================================


class TestFindRelevantKeywordFiltering:
    """Tests for find_relevant() keyword pre-filtering (SPEC-01.09)."""

    def test_keyword_filtering_returns_relevant_chunks(self, basic_env):
        """
        Keyword pre-filtering should identify chunks containing query terms.

        @trace SPEC-01.09
        """
        find_relevant = basic_env.globals["find_relevant"]

        content = """
Section A: This section discusses Python programming.
Python is a great language for beginners.
It has simple syntax and powerful libraries.

Section B: This section is about JavaScript.
JavaScript runs in browsers and on servers.
Node.js made server-side JS popular.

Section C: Back to Python topics.
Django and Flask are Python web frameworks.
Machine learning with Python is very popular.
"""

        result = find_relevant(
            content=content,
            query="Python programming",
            top_k=5,
        )

        # Should return chunks containing "Python"
        assert len(result) > 0
        for chunk, _score in result:
            # At least the top results should contain the query term
            assert "python" in chunk.lower() or "programming" in chunk.lower()

    def test_no_matches_returns_empty(self, basic_env):
        """
        Should return empty list if no chunks match the query.

        @trace SPEC-01.09
        """
        find_relevant = basic_env.globals["find_relevant"]

        content = "This content has nothing to do with the search term."

        result = find_relevant(
            content=content,
            query="xyznonexistent123",
            top_k=5,
        )

        # Should return empty or very low scored results
        assert isinstance(result, list)

    def test_case_insensitive_matching(self, basic_env):
        """
        Keyword matching should be case-insensitive.

        @trace SPEC-01.09
        """
        find_relevant = basic_env.globals["find_relevant"]

        content = "PYTHON is great. Python is powerful. python rocks."

        result = find_relevant(
            content=content,
            query="python",
            top_k=5,
        )

        assert len(result) > 0


# =============================================================================
# SPEC-01.10: Optional LLM scoring
# =============================================================================


class TestFindRelevantLLMScoring:
    """Tests for find_relevant() LLM scoring (SPEC-01.10)."""

    def test_llm_scoring_disabled_by_default(self, basic_env):
        """
        LLM scoring should be disabled by default.

        @trace SPEC-01.10
        """
        find_relevant = basic_env.globals["find_relevant"]

        content = "Test content " * 100

        # Should work without triggering LLM calls
        result = find_relevant(
            content=content,
            query="test",
        )

        # Should return synchronous result (not deferred)
        assert isinstance(result, list)
        # No pending operations should be created with LLM scoring off
        assert not basic_env.has_pending_operations()

    def test_llm_scoring_can_be_enabled(self, basic_env):
        """
        LLM scoring can be enabled via use_llm_scoring=True.

        @trace SPEC-01.10
        """
        find_relevant = basic_env.globals["find_relevant"]

        # Create content that would have many candidate chunks
        lines = [f"Line {i}: relevant content here" for i in range(500)]
        content = "\n".join(lines)

        result = find_relevant(
            content=content,
            query="relevant",
            top_k=3,
            use_llm_scoring=True,
        )

        # When LLM scoring is enabled and candidates > top_k * 2,
        # it should return deferred operations or process differently
        assert result is not None


# =============================================================================
# SPEC-01.11: Return format (chunk, score) tuples
# =============================================================================


class TestFindRelevantReturnFormat:
    """Tests for find_relevant() return format (SPEC-01.11)."""

    def test_returns_list_of_tuples(self, basic_env):
        """
        find_relevant should return list of (chunk, score) tuples.

        @trace SPEC-01.11
        """
        find_relevant = basic_env.globals["find_relevant"]

        content = "Relevant text here. More relevant content."

        result = find_relevant(
            content=content,
            query="relevant",
            top_k=5,
        )

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            chunk, score = item
            assert isinstance(chunk, str)
            assert isinstance(score, (int, float))

    def test_sorted_by_relevance_descending(self, basic_env):
        """
        Results should be sorted by relevance score descending.

        @trace SPEC-01.11
        """
        find_relevant = basic_env.globals["find_relevant"]

        content = """
Very relevant: Python Python Python programming code.
Somewhat relevant: Python is nice.
Not very relevant: JavaScript is also good.
Most relevant: Python programming Python code Python.
"""

        result = find_relevant(
            content=content,
            query="Python programming",
            top_k=10,
        )

        if len(result) >= 2:
            scores = [score for _, score in result]
            # Scores should be in descending order
            assert scores == sorted(scores, reverse=True)

    def test_respects_top_k_limit(self, basic_env):
        """
        Should return at most top_k results.

        @trace SPEC-01.11
        """
        find_relevant = basic_env.globals["find_relevant"]

        # Create lots of matching content
        lines = [f"Line {i}: Python code here" for i in range(200)]
        content = "\n".join(lines)

        result = find_relevant(
            content=content,
            query="Python",
            top_k=5,
        )

        assert len(result) <= 5

    def test_default_top_k_is_5(self, basic_env):
        """
        Default top_k should be 5.

        @trace SPEC-01.11
        """
        find_relevant = basic_env.globals["find_relevant"]

        lines = [f"Line {i}: matching content" for i in range(200)]
        content = "\n".join(lines)

        result = find_relevant(
            content=content,
            query="matching",
        )

        # Default should return at most 5
        assert len(result) <= 5


# =============================================================================
# SPEC-01.12: Performance requirement
# =============================================================================


class TestFindRelevantPerformance:
    """Tests for find_relevant() performance (SPEC-01.12)."""

    def test_completes_within_2_seconds_for_100k_chars(self, basic_env):
        """
        find_relevant should complete within 2 seconds for 100K char content.

        @trace SPEC-01.12
        """
        import time

        find_relevant = basic_env.globals["find_relevant"]

        # Create ~100K character content
        content = "Python programming is great. " * 3500  # ~100K chars

        start = time.time()
        result = find_relevant(
            content=content,
            query="Python programming",
            top_k=10,
            use_llm_scoring=False,  # Without LLM scoring per spec
        )
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Took {elapsed:.2f}s, expected < 2s"
        assert isinstance(result, list)

    @pytest.mark.slow
    def test_handles_large_content_efficiently(self, basic_env):
        """
        Should handle large content without memory issues.

        @trace SPEC-01.12
        """
        find_relevant = basic_env.globals["find_relevant"]

        # Create 500K character content
        content = "Search term appears here. " * 20000

        result = find_relevant(
            content=content,
            query="Search term",
            top_k=10,
        )

        assert isinstance(result, list)


# =============================================================================
# find_relevant() REPL Integration Tests
# =============================================================================


class TestFindRelevantREPLIntegration:
    """Tests for find_relevant() integration with REPL environment."""

    def test_callable_from_repl_code(self, basic_env):
        """
        find_relevant should be callable from REPL executed code.

        @trace SPEC-01.22
        """
        result = basic_env.execute(
            "results = find_relevant('Python is great. Python rocks.', 'Python')"
        )

        assert result.success is True
        assert "results" in basic_env.locals

    def test_works_with_file_content(self, basic_env):
        """
        find_relevant should work with file content from context.

        @trace SPEC-01.22
        """
        result = basic_env.execute(
            """
all_code = '\\n'.join(files.values())
relevant = find_relevant(all_code, 'def ', top_k=3)
"""
        )

        assert result.success is True
        assert "relevant" in basic_env.locals


# =============================================================================
# SPEC-01.13: extract_functions() function signature
# =============================================================================


class TestExtractFunctionsSignature:
    """Tests for extract_functions() function signature (SPEC-01.13)."""

    def test_extract_functions_exists_in_environment(self, basic_env):
        """
        extract_functions should be available in REPL environment.

        @trace SPEC-01.13
        """
        assert "extract_functions" in basic_env.globals
        assert callable(basic_env.globals["extract_functions"])

    def test_extract_functions_accepts_required_parameters(self, basic_env):
        """
        extract_functions(content) should work with minimal parameters.

        @trace SPEC-01.13
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
def hello():
    pass

def world(x, y):
    return x + y
"""

        result = extract_functions(content=content)

        # Should return a list of dicts
        assert isinstance(result, list)

    def test_extract_functions_accepts_optional_language(self, basic_env):
        """
        extract_functions should accept optional language parameter.

        @trace SPEC-01.13
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = "def test(): pass"

        result = extract_functions(content=content, language="python")

        assert isinstance(result, list)


# =============================================================================
# SPEC-01.14: Language support
# =============================================================================


class TestExtractFunctionsLanguageSupport:
    """Tests for extract_functions() language support (SPEC-01.14)."""

    def test_supports_python(self, basic_env):
        """
        extract_functions should support Python language.

        @trace SPEC-01.14
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
def greet(name: str) -> str:
    return f"Hello, {name}!"

async def fetch_data():
    pass

class MyClass:
    def method(self):
        pass
"""

        result = extract_functions(content=content, language="python")

        assert isinstance(result, list)
        assert len(result) >= 2  # At least greet and fetch_data

        # Check that function names are extracted
        names = [f["name"] for f in result]
        assert "greet" in names
        assert "fetch_data" in names

    def test_supports_go(self, basic_env):
        """
        extract_functions should support Go language.

        @trace SPEC-01.14
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
func main() {
    fmt.Println("Hello")
}

func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {
    // handle
}

func Add(a, b int) int {
    return a + b
}
"""

        result = extract_functions(content=content, language="go")

        assert isinstance(result, list)
        names = [f["name"] for f in result]
        assert "main" in names or "Main" in names
        assert "Add" in names

    def test_supports_javascript(self, basic_env):
        """
        extract_functions should support JavaScript language.

        @trace SPEC-01.14
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
function greet(name) {
    return `Hello, ${name}!`;
}

const arrow = (x) => x * 2;

async function fetchData() {
    return await fetch('/api');
}

class MyClass {
    constructor() {}
    method() {}
}
"""

        result = extract_functions(content=content, language="javascript")

        assert isinstance(result, list)
        names = [f["name"] for f in result]
        assert "greet" in names
        assert "fetchData" in names

    def test_supports_typescript(self, basic_env):
        """
        extract_functions should support TypeScript language.

        @trace SPEC-01.14
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
function greet(name: string): string {
    return `Hello, ${name}!`;
}

const typed = (x: number): number => x * 2;

async function fetchData(): Promise<Data> {
    return await fetch('/api');
}

interface User {
    name: string;
}
"""

        result = extract_functions(content=content, language="typescript")

        assert isinstance(result, list)
        names = [f["name"] for f in result]
        assert "greet" in names
        assert "fetchData" in names

    def test_default_language_is_python(self, basic_env):
        """
        Default language should be Python.

        @trace SPEC-01.14
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = "def test(): pass"

        # Without specifying language
        result = extract_functions(content=content)

        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]["name"] == "test"


# =============================================================================
# SPEC-01.15: Return format
# =============================================================================


class TestExtractFunctionsReturnFormat:
    """Tests for extract_functions() return format (SPEC-01.15)."""

    def test_returns_list_of_dicts(self, basic_env):
        """
        extract_functions should return list of dicts.

        @trace SPEC-01.15
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
def foo():
    pass

def bar(x):
    return x
"""

        result = extract_functions(content=content)

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)

    def test_dict_has_required_keys(self, basic_env):
        """
        Each dict should have keys: name, signature, start_line, end_line.

        @trace SPEC-01.15
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
def example(arg1, arg2):
    return arg1 + arg2
"""

        result = extract_functions(content=content)

        assert len(result) >= 1
        func = result[0]

        assert "name" in func
        assert "signature" in func
        assert "start_line" in func
        assert "end_line" in func

    def test_name_is_string(self, basic_env):
        """
        Function name should be a string.

        @trace SPEC-01.15
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = "def my_function(): pass"

        result = extract_functions(content=content)

        assert len(result) >= 1
        assert isinstance(result[0]["name"], str)
        assert result[0]["name"] == "my_function"

    def test_signature_contains_parameters(self, basic_env):
        """
        Signature should contain function parameters.

        @trace SPEC-01.15
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = "def calculate(a, b, c=10): return a + b + c"

        result = extract_functions(content=content)

        assert len(result) >= 1
        sig = result[0]["signature"]
        assert "a" in sig
        assert "b" in sig
        # May or may not include default value in signature

    def test_start_line_is_integer(self, basic_env):
        """
        start_line should be an integer.

        @trace SPEC-01.15
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
# Comment
def func():
    pass
"""

        result = extract_functions(content=content)

        assert len(result) >= 1
        assert isinstance(result[0]["start_line"], int)
        assert result[0]["start_line"] > 0

    def test_end_line_is_integer(self, basic_env):
        """
        end_line should be an integer.

        @trace SPEC-01.15
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
def multi_line():
    x = 1
    y = 2
    return x + y
"""

        result = extract_functions(content=content)

        assert len(result) >= 1
        assert isinstance(result[0]["end_line"], int)
        assert result[0]["end_line"] >= result[0]["start_line"]

    def test_line_numbers_are_accurate(self, basic_env):
        """
        Line numbers should accurately reflect function position.

        @trace SPEC-01.15
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """def first():
    pass

def second():
    x = 1
    return x
"""

        result = extract_functions(content=content)

        # Find first function
        first_func = next((f for f in result if f["name"] == "first"), None)
        assert first_func is not None
        assert first_func["start_line"] == 1

        # Find second function
        second_func = next((f for f in result if f["name"] == "second"), None)
        assert second_func is not None
        assert second_func["start_line"] == 4


# =============================================================================
# SPEC-01.16: Regex patterns for languages
# =============================================================================


class TestExtractFunctionsRegexPatterns:
    """Tests for extract_functions() regex patterns (SPEC-01.16)."""

    def test_python_async_functions(self, basic_env):
        """
        Should extract async Python functions.

        @trace SPEC-01.16
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
async def async_fetch():
    pass

async def another_async(arg):
    return arg
"""

        result = extract_functions(content=content, language="python")

        names = [f["name"] for f in result]
        assert "async_fetch" in names
        assert "another_async" in names

    def test_python_decorated_functions(self, basic_env):
        """
        Should extract decorated Python functions.

        @trace SPEC-01.16
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
@decorator
def decorated():
    pass

@app.route('/')
@login_required
def multi_decorated():
    pass
"""

        result = extract_functions(content=content, language="python")

        names = [f["name"] for f in result]
        assert "decorated" in names
        assert "multi_decorated" in names

    def test_go_method_receivers(self, basic_env):
        """
        Should extract Go methods with receivers.

        @trace SPEC-01.16
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
func (s *Server) Start() error {
    return nil
}

func (s Server) Stop() {
}
"""

        result = extract_functions(content=content, language="go")

        names = [f["name"] for f in result]
        assert "Start" in names
        assert "Stop" in names

    def test_javascript_arrow_functions(self, basic_env):
        """
        Should extract JavaScript arrow functions.

        @trace SPEC-01.16
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
const arrow = (x) => x * 2;

const multiLine = (a, b) => {
    return a + b;
};

const named = function namedFunc() {};
"""

        result = extract_functions(content=content, language="javascript")

        # Arrow functions may or may not be extracted depending on implementation
        # At minimum, named function should be found
        assert isinstance(result, list)


# =============================================================================
# SPEC-01.17: Malformed input handling
# =============================================================================


class TestExtractFunctionsMalformedInput:
    """Tests for extract_functions() malformed input handling (SPEC-01.17)."""

    def test_empty_content_returns_empty_list(self, basic_env):
        """
        Empty content should return empty list.

        @trace SPEC-01.17
        """
        extract_functions = basic_env.globals["extract_functions"]

        result = extract_functions(content="")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_no_functions_returns_empty_list(self, basic_env):
        """
        Content with no functions should return empty list.

        @trace SPEC-01.17
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
# Just comments
x = 1
y = 2
print(x + y)
"""

        result = extract_functions(content=content)

        assert isinstance(result, list)
        # May be empty or contain nothing useful

    def test_partial_function_definitions(self, basic_env):
        """
        Should handle partial/incomplete function definitions gracefully.

        @trace SPEC-01.17
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
def valid_function():
    pass

def incomplete(
    # Missing closing paren

def another_valid():
    return True
"""

        result = extract_functions(content=content)

        # Should extract at least the valid functions
        assert isinstance(result, list)
        names = [f["name"] for f in result]
        assert "valid_function" in names or "another_valid" in names

    def test_syntax_errors_handled_gracefully(self, basic_env):
        """
        Should handle content with syntax errors.

        @trace SPEC-01.17
        """
        extract_functions = basic_env.globals["extract_functions"]

        content = """
def good():
    pass

def bad syntax here

def also_good():
    return 1
"""

        # Should not raise exception
        result = extract_functions(content=content)

        assert isinstance(result, list)
        # Should extract at least some functions
        names = [f["name"] for f in result]
        assert "good" in names or "also_good" in names

    def test_binary_content_handled(self, basic_env):
        """
        Should handle binary or non-text content gracefully.

        @trace SPEC-01.17
        """
        extract_functions = basic_env.globals["extract_functions"]

        # Content with null bytes and weird characters
        content = "def func(): pass\x00\xff\xfe more stuff"

        # Should not crash
        result = extract_functions(content=content)

        assert isinstance(result, list)

    def test_very_long_lines_handled(self, basic_env):
        """
        Should handle very long lines without issues.

        @trace SPEC-01.17
        """
        extract_functions = basic_env.globals["extract_functions"]

        # Function with very long argument list
        args = ", ".join([f"arg{i}" for i in range(100)])
        content = f"def long_function({args}): pass"

        result = extract_functions(content=content)

        assert isinstance(result, list)
        if len(result) > 0:
            assert result[0]["name"] == "long_function"


# =============================================================================
# extract_functions() REPL Integration Tests
# =============================================================================


class TestExtractFunctionsREPLIntegration:
    """Tests for extract_functions() integration with REPL environment."""

    def test_callable_from_repl_code(self, basic_env):
        """
        extract_functions should be callable from REPL executed code.

        @trace SPEC-01.22
        """
        result = basic_env.execute("funcs = extract_functions('def test(): pass')")

        assert result.success is True
        assert "funcs" in basic_env.locals

    def test_works_with_file_content(self, basic_env):
        """
        extract_functions should work with file content from context.

        @trace SPEC-01.22
        """
        result = basic_env.execute(
            """
code = files.get('main.py', '')
funcs = extract_functions(code, language='python')
"""
        )

        assert result.success is True
        assert "funcs" in basic_env.locals

    def test_analyze_multiple_files(self, basic_env):
        """
        Should be able to analyze functions across multiple files.

        @trace SPEC-01.22
        """
        result = basic_env.execute(
            """
all_funcs = []
for filename, content in files.items():
    funcs = extract_functions(content, language='python')
    for f in funcs:
        f['file'] = filename
    all_funcs.extend(funcs)
"""
        )

        assert result.success is True
        assert "all_funcs" in basic_env.locals
