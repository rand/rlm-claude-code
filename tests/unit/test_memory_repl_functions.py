"""
Unit tests for REPL memory functions.

Implements: Spec SPEC-02.27-34 tests for REPL integration and security.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.types import Message, MessageRole, SessionContext

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)
    for suffix in ["-wal", "-shm"]:
        wal_path = path + suffix
        if os.path.exists(wal_path):
            os.unlink(wal_path)


@pytest.fixture
def basic_context():
    """Create a basic context for testing."""
    return SessionContext(
        messages=[Message(role=MessageRole.USER, content="Test message")],
        files={},
        tool_outputs=[],
        working_memory={},
    )


@pytest.fixture
def memory_repl_env(basic_context, temp_db_path):
    """Create REPL environment with memory functions enabled."""
    from src.memory_store import MemoryStore
    from src.repl_environment import RLMEnvironment

    # Create memory store
    store = MemoryStore(db_path=temp_db_path)

    # Create REPL environment with memory integration
    env = RLMEnvironment(basic_context, use_restricted=False)
    env.enable_memory(store)

    return env


# =============================================================================
# SPEC-02.27: memory_query()
# =============================================================================


class TestMemoryQuery:
    """Tests for memory_query REPL function."""

    def test_memory_query_exists_in_environment(self, memory_repl_env):
        """
        memory_query should be available in REPL environment.

        @trace SPEC-02.27
        """
        assert "memory_query" in memory_repl_env.globals
        assert callable(memory_repl_env.globals["memory_query"])

    def test_memory_query_accepts_query_string(self, memory_repl_env):
        """
        memory_query should accept a query string.

        @trace SPEC-02.27
        """
        memory_query = memory_repl_env.globals["memory_query"]

        # Should not raise
        result = memory_query("test query")
        assert isinstance(result, list)

    def test_memory_query_accepts_limit(self, memory_repl_env):
        """
        memory_query should accept optional limit parameter.

        @trace SPEC-02.27
        """
        memory_query = memory_repl_env.globals["memory_query"]
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        # Add multiple facts
        for i in range(10):
            memory_add_fact(f"Fact number {i}", confidence=0.8)

        # Query with limit
        results = memory_query("fact", limit=3)
        assert len(results) <= 3

    def test_memory_query_returns_relevant_nodes(self, memory_repl_env):
        """
        memory_query should return nodes matching the query.

        @trace SPEC-02.27
        """
        memory_query = memory_repl_env.globals["memory_query"]
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        # Add facts with specific content
        memory_add_fact("Python is a programming language", confidence=0.9)
        memory_add_fact("JavaScript is also a language", confidence=0.9)
        memory_add_fact("Cats are cute animals", confidence=0.9)

        # Query for programming-related
        results = memory_query("programming")

        assert len(results) >= 1
        # Results should contain Python fact
        contents = [r.content for r in results]
        assert any("Python" in c for c in contents)

    def test_memory_query_empty_returns_empty_list(self, memory_repl_env):
        """
        memory_query with no matches should return empty list.

        @trace SPEC-02.27
        """
        memory_query = memory_repl_env.globals["memory_query"]

        results = memory_query("nonexistent_term_xyz123")
        assert results == []


# =============================================================================
# SPEC-02.28: memory_add_fact()
# =============================================================================


class TestMemoryAddFact:
    """Tests for memory_add_fact REPL function."""

    def test_memory_add_fact_exists_in_environment(self, memory_repl_env):
        """
        memory_add_fact should be available in REPL environment.

        @trace SPEC-02.28
        """
        assert "memory_add_fact" in memory_repl_env.globals
        assert callable(memory_repl_env.globals["memory_add_fact"])

    def test_memory_add_fact_creates_node(self, memory_repl_env):
        """
        memory_add_fact should create a fact node.

        @trace SPEC-02.28
        """
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]
        memory_query = memory_repl_env.globals["memory_query"]

        # Add fact
        node_id = memory_add_fact("The sky is blue", confidence=0.95)

        assert node_id is not None
        assert isinstance(node_id, str)

        # Should be queryable
        results = memory_query("sky blue")
        assert len(results) >= 1

    def test_memory_add_fact_accepts_confidence(self, memory_repl_env):
        """
        memory_add_fact should accept confidence parameter.

        @trace SPEC-02.28
        """
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        node_id = memory_add_fact("Test fact", confidence=0.7)

        # Get the node and verify confidence
        store = memory_repl_env._memory_store
        node = store.get_node(node_id)
        assert node.confidence == 0.7

    def test_memory_add_fact_default_confidence(self, memory_repl_env):
        """
        memory_add_fact should have default confidence.

        @trace SPEC-02.28
        """
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        node_id = memory_add_fact("Test fact without confidence")

        store = memory_repl_env._memory_store
        node = store.get_node(node_id)
        assert node.confidence == 0.5  # Default

    def test_memory_add_fact_sets_correct_type(self, memory_repl_env):
        """
        memory_add_fact should create node with type 'fact'.

        @trace SPEC-02.28
        """
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        node_id = memory_add_fact("Test fact")

        store = memory_repl_env._memory_store
        node = store.get_node(node_id)
        assert node.type == "fact"


# =============================================================================
# SPEC-02.29: memory_add_experience()
# =============================================================================


class TestMemoryAddExperience:
    """Tests for memory_add_experience REPL function."""

    def test_memory_add_experience_exists_in_environment(self, memory_repl_env):
        """
        memory_add_experience should be available in REPL environment.

        @trace SPEC-02.29
        """
        assert "memory_add_experience" in memory_repl_env.globals
        assert callable(memory_repl_env.globals["memory_add_experience"])

    def test_memory_add_experience_creates_node(self, memory_repl_env):
        """
        memory_add_experience should create an experience node.

        @trace SPEC-02.29
        """
        memory_add_experience = memory_repl_env.globals["memory_add_experience"]

        node_id = memory_add_experience(
            content="Debugged TypeError in auth module",
            outcome="Fixed by adding type check",
            success=True,
        )

        assert node_id is not None

        store = memory_repl_env._memory_store
        node = store.get_node(node_id)
        assert node.type == "experience"

    def test_memory_add_experience_stores_outcome(self, memory_repl_env):
        """
        memory_add_experience should store outcome in metadata.

        @trace SPEC-02.29
        """
        memory_add_experience = memory_repl_env.globals["memory_add_experience"]

        node_id = memory_add_experience(
            content="Attempted optimization",
            outcome="Reduced latency by 50%",
            success=True,
        )

        store = memory_repl_env._memory_store
        node = store.get_node(node_id)
        assert "outcome" in node.metadata
        assert node.metadata["outcome"] == "Reduced latency by 50%"

    def test_memory_add_experience_stores_success(self, memory_repl_env):
        """
        memory_add_experience should store success flag.

        @trace SPEC-02.29
        """
        memory_add_experience = memory_repl_env.globals["memory_add_experience"]

        # Successful experience
        node_id1 = memory_add_experience(content="Good experience", outcome="Worked", success=True)
        # Failed experience
        node_id2 = memory_add_experience(content="Bad experience", outcome="Failed", success=False)

        store = memory_repl_env._memory_store

        node1 = store.get_node(node_id1)
        assert node1.metadata.get("success") is True

        node2 = store.get_node(node_id2)
        assert node2.metadata.get("success") is False


# =============================================================================
# SPEC-02.30: memory_get_context()
# =============================================================================


class TestMemoryGetContext:
    """Tests for memory_get_context REPL function."""

    def test_memory_get_context_exists_in_environment(self, memory_repl_env):
        """
        memory_get_context should be available in REPL environment.

        @trace SPEC-02.30
        """
        assert "memory_get_context" in memory_repl_env.globals
        assert callable(memory_repl_env.globals["memory_get_context"])

    def test_memory_get_context_returns_recent_nodes(self, memory_repl_env):
        """
        memory_get_context should return recent/relevant nodes.

        @trace SPEC-02.30
        """
        memory_get_context = memory_repl_env.globals["memory_get_context"]
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        # Add some facts
        memory_add_fact("Fact 1", confidence=0.9)
        memory_add_fact("Fact 2", confidence=0.8)
        memory_add_fact("Fact 3", confidence=0.7)

        context = memory_get_context()

        assert isinstance(context, list)
        assert len(context) >= 1

    def test_memory_get_context_accepts_limit(self, memory_repl_env):
        """
        memory_get_context should accept limit parameter.

        @trace SPEC-02.30
        """
        memory_get_context = memory_repl_env.globals["memory_get_context"]
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        # Add many facts
        for i in range(10):
            memory_add_fact(f"Context fact {i}", confidence=0.8)

        context = memory_get_context(limit=5)

        assert len(context) <= 5

    def test_memory_get_context_prioritizes_high_confidence(self, memory_repl_env):
        """
        memory_get_context should prioritize high confidence nodes.

        @trace SPEC-02.30
        """
        memory_get_context = memory_repl_env.globals["memory_get_context"]
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        # Add facts with different confidence
        memory_add_fact("Low confidence fact", confidence=0.3)
        memory_add_fact("High confidence fact", confidence=0.95)

        context = memory_get_context(limit=1)

        assert len(context) == 1
        assert context[0].confidence >= 0.9


# =============================================================================
# SPEC-02.31: memory_relate()
# =============================================================================


class TestMemoryRelate:
    """Tests for memory_relate REPL function."""

    def test_memory_relate_exists_in_environment(self, memory_repl_env):
        """
        memory_relate should be available in REPL environment.

        @trace SPEC-02.31
        """
        assert "memory_relate" in memory_repl_env.globals
        assert callable(memory_repl_env.globals["memory_relate"])

    def test_memory_relate_creates_edge(self, memory_repl_env):
        """
        memory_relate should create an edge between nodes.

        @trace SPEC-02.31
        """
        memory_relate = memory_repl_env.globals["memory_relate"]
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        # Create two facts
        fact1 = memory_add_fact("Python is a language")
        fact2 = memory_add_fact("Python is interpreted")

        # Relate them
        edge_id = memory_relate("implies", fact1, fact2)

        assert edge_id is not None
        assert isinstance(edge_id, str)

    def test_memory_relate_with_label(self, memory_repl_env):
        """
        memory_relate should accept a label parameter.

        @trace SPEC-02.31
        """
        memory_relate = memory_repl_env.globals["memory_relate"]
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        fact1 = memory_add_fact("Concept A")
        fact2 = memory_add_fact("Concept B")

        edge_id = memory_relate("causes", fact1, fact2)

        store = memory_repl_env._memory_store
        edge = store.get_edge(edge_id)
        assert edge.label == "causes"

    def test_memory_relate_makes_nodes_related(self, memory_repl_env):
        """
        After memory_relate, nodes should be found as related.

        @trace SPEC-02.31
        """
        memory_relate = memory_repl_env.globals["memory_relate"]
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]

        fact1 = memory_add_fact("Fact A")
        fact2 = memory_add_fact("Fact B")

        memory_relate("connects", fact1, fact2)

        store = memory_repl_env._memory_store
        related = store.get_related_nodes(fact1)
        related_ids = [n.id for n in related]

        assert fact2 in related_ids


# =============================================================================
# SPEC-02.32-34: Security
# =============================================================================


class TestMemoryREPLSecurity:
    """Tests for memory function security in REPL."""

    def test_functions_work_in_restricted_mode(self, basic_context, temp_db_path):
        """
        Memory functions should work within RestrictedPython sandbox.

        @trace SPEC-02.32
        """
        from src.memory_store import MemoryStore
        from src.repl_environment import RLMEnvironment

        store = MemoryStore(db_path=temp_db_path)
        env = RLMEnvironment(basic_context, use_restricted=True)
        env.enable_memory(store)

        # Execute memory operations via REPL
        result = env.execute("""
node_id = memory_add_fact("Test fact", confidence=0.8)
_ = node_id
""")

        assert result.success, f"Execution failed: {result.error}"
        assert result.output is not None

    def test_no_arbitrary_sql_execution(self, memory_repl_env):
        """
        Memory operations should not allow arbitrary SQL.

        @trace SPEC-02.33
        """
        # Memory functions should not have execute_sql or similar
        assert "execute_sql" not in memory_repl_env.globals
        assert "raw_query" not in memory_repl_env.globals

        # Store should not be directly accessible for raw queries
        store = memory_repl_env._memory_store
        assert not hasattr(store, "execute_raw") or not callable(
            getattr(store, "execute_raw", None)
        )

    def test_content_sanitization(self, memory_repl_env):
        """
        Node content should be sanitized.

        @trace SPEC-02.34
        """
        memory_add_fact = memory_repl_env.globals["memory_add_fact"]
        memory_query = memory_repl_env.globals["memory_query"]

        # Attempt SQL injection
        malicious_content = "Test'; DROP TABLE nodes; --"
        node_id = memory_add_fact(malicious_content)

        # Table should still exist and work
        store = memory_repl_env._memory_store
        node = store.get_node(node_id)
        assert node is not None
        assert node.content == malicious_content

        # Query should still work
        results = memory_query("test")
        assert len(results) >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryREPLIntegration:
    """Integration tests for memory REPL functions."""

    def test_full_workflow(self, memory_repl_env):
        """
        Test complete memory workflow via REPL.

        @trace SPEC-02.27-31
        """
        env = memory_repl_env

        # Add facts
        result = env.execute("""
fact1 = memory_add_fact("Python supports OOP", confidence=0.9)
fact2 = memory_add_fact("Python has classes", confidence=0.85)
fact3 = memory_add_fact("Classes enable OOP", confidence=0.95)
""")
        assert result.success

        # Create relations
        result = env.execute("""
memory_relate("supports", fact1, fact2)
memory_relate("enables", fact2, fact3)
""")
        assert result.success

        # Query
        result = env.execute("""
results = memory_query("Python OOP")
_ = len(results)
""")
        assert result.success
        assert result.output >= 1

        # Get context
        result = env.execute("""
context = memory_get_context(limit=10)
_ = len(context)
""")
        assert result.success

    def test_experience_workflow(self, memory_repl_env):
        """
        Test experience recording workflow.

        @trace SPEC-02.29
        """
        env = memory_repl_env

        result = env.execute("""
# Record a debugging experience
exp_id = memory_add_experience(
    content="Debugged ImportError in module X",
    outcome="Fixed by adding __init__.py",
    success=True
)

# Query experiences
context = memory_get_context(limit=5)
has_experience = any(n.type == 'experience' for n in context)
_ = has_experience
""")

        assert result.success
        assert result.output is True


# =============================================================================
# Default State Tests
# =============================================================================


class TestMemoryREPLDefaults:
    """Tests for default state when memory is not enabled."""

    def test_functions_not_available_by_default(self, basic_context):
        """
        Memory functions should not be available without enable_memory().

        @trace SPEC-02.27-31
        """
        from src.repl_environment import RLMEnvironment

        env = RLMEnvironment(basic_context, use_restricted=False)

        # Memory functions should not be in globals
        assert "memory_query" not in env.globals
        assert "memory_add_fact" not in env.globals
        assert "memory_add_experience" not in env.globals
        assert "memory_get_context" not in env.globals
        assert "memory_relate" not in env.globals

    def test_enable_memory_adds_functions(self, basic_context, temp_db_path):
        """
        enable_memory() should add memory functions to globals.

        @trace SPEC-02.27-31
        """
        from src.memory_store import MemoryStore
        from src.repl_environment import RLMEnvironment

        env = RLMEnvironment(basic_context, use_restricted=False)
        store = MemoryStore(db_path=temp_db_path)

        # Before enabling
        assert "memory_query" not in env.globals

        # Enable memory
        env.enable_memory(store)

        # After enabling
        assert "memory_query" in env.globals
        assert "memory_add_fact" in env.globals
        assert "memory_add_experience" in env.globals
        assert "memory_get_context" in env.globals
        assert "memory_relate" in env.globals
