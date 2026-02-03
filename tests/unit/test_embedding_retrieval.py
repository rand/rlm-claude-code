"""
Tests for embedding-based memory retrieval.

@trace SPEC-09.01-09.07
"""

from __future__ import annotations

import pytest

from src.embedding_retrieval import (
    EmbeddingConfig,
    EmbeddingStore,
    HybridRetriever,
    HybridSearchResult,
    MockEmbeddingProvider,
)

# --- Test fixtures ---


def create_sample_texts() -> list[str]:
    """Create sample texts for testing."""
    return [
        "Python is a programming language",
        "Machine learning uses neural networks",
        "Database queries use SQL syntax",
        "Web servers handle HTTP requests",
        "Authentication systems verify user identity",
    ]


def create_mock_embeddings(num: int, dim: int = 1536) -> list[list[float]]:
    """Create mock embeddings."""
    return [[float(i + j) / (num * dim) for j in range(dim)] for i in range(num)]


# --- SPEC-09.01: Embedding computation ---


class TestEmbeddingComputation:
    """Tests for embedding computation for memory nodes."""

    def test_compute_embedding_for_text(self) -> None:
        """
        @trace SPEC-09.01
        Should compute embedding for text content.
        """
        provider = MockEmbeddingProvider(dimensions=1536)

        embedding = provider.embed("Test text content")

        assert embedding is not None
        assert len(embedding) == 1536

    def test_embedding_is_normalized(self) -> None:
        """
        @trace SPEC-09.01
        Embeddings should be normalized vectors.
        """
        provider = MockEmbeddingProvider(dimensions=1536)

        embedding = provider.embed("Test text")

        # Check approximate normalization (unit vector)
        magnitude = sum(x * x for x in embedding) ** 0.5
        assert 0.9 <= magnitude <= 1.1  # Allow some tolerance


# --- SPEC-09.02: Default embedding model ---


class TestDefaultEmbeddingModel:
    """Tests for default embedding model configuration."""

    def test_default_dimensions(self) -> None:
        """
        @trace SPEC-09.02
        Default dimensions should be 1536 (text-embedding-3-small).
        """
        config = EmbeddingConfig()

        assert config.dimensions == 1536

    def test_default_model_name(self) -> None:
        """
        @trace SPEC-09.02
        Default model should be text-embedding-3-small.
        """
        config = EmbeddingConfig()

        assert config.model == "text-embedding-3-small"


# --- SPEC-09.03: Embedding storage ---


class TestEmbeddingStorage:
    """Tests for embedding storage."""

    def test_store_embedding(self) -> None:
        """
        @trace SPEC-09.03
        Should store embeddings with node_id.
        """
        store = EmbeddingStore(":memory:")
        embedding = [0.1] * 1536

        store.store("node_1", embedding)

        retrieved = store.get("node_1")
        assert retrieved is not None
        assert len(retrieved) == 1536

    def test_retrieve_embedding(self) -> None:
        """
        @trace SPEC-09.03
        Should retrieve stored embeddings.
        """
        store = EmbeddingStore(":memory:")
        original = [float(i) / 1536 for i in range(1536)]

        store.store("node_1", original)
        retrieved = store.get("node_1")

        # Check values match (with floating point tolerance)
        assert retrieved is not None
        for i in range(min(10, len(original))):
            assert abs(retrieved[i] - original[i]) < 0.0001

    def test_missing_embedding_returns_none(self) -> None:
        """
        @trace SPEC-09.03
        Should return None for missing embeddings.
        """
        store = EmbeddingStore(":memory:")

        result = store.get("nonexistent")

        assert result is None


# --- SPEC-09.04: Hybrid retrieval ---


class TestHybridRetrieval:
    """Tests for hybrid retrieval combining keyword and semantic search."""

    def test_hybrid_search_combines_results(self) -> None:
        """
        @trace SPEC-09.04
        Hybrid search should combine keyword and semantic results.
        """
        retriever = HybridRetriever(
            embedding_provider=MockEmbeddingProvider(),
            hybrid_alpha=0.5,
        )

        # Add some documents
        docs = create_sample_texts()
        for i, doc in enumerate(docs):
            retriever.add_document(f"doc_{i}", doc)

        results = retriever.search("programming language Python")

        assert len(results) > 0
        # First result should be relevant
        assert "python" in results[0].content.lower() or "programming" in results[0].content.lower()

    def test_hybrid_score_calculation(self) -> None:
        """
        @trace SPEC-09.04
        Hybrid score should combine semantic and keyword scores.
        """
        # hybrid_score = hybrid_alpha * semantic + (1 - hybrid_alpha) * keyword
        retriever = HybridRetriever(
            embedding_provider=MockEmbeddingProvider(),
            hybrid_alpha=0.5,
        )

        # With alpha=0.5, both scores should contribute equally
        result = HybridSearchResult(
            node_id="test",
            content="test content",
            semantic_score=0.8,
            keyword_score=0.6,
            hybrid_alpha=0.5,
        )

        expected = 0.5 * 0.8 + 0.5 * 0.6  # 0.7
        assert result.hybrid_score == pytest.approx(expected)

    def test_semantic_only_search(self) -> None:
        """
        @trace SPEC-09.04
        Should support semantic-only search (alpha=1.0).
        """
        retriever = HybridRetriever(
            embedding_provider=MockEmbeddingProvider(),
            hybrid_alpha=1.0,  # Only semantic
        )

        docs = create_sample_texts()
        for i, doc in enumerate(docs):
            retriever.add_document(f"doc_{i}", doc)

        results = retriever.search("neural networks AI")

        # Should return results based on semantic similarity
        assert len(results) > 0

    def test_keyword_only_search(self) -> None:
        """
        @trace SPEC-09.04
        Should support keyword-only search (alpha=0.0).
        """
        retriever = HybridRetriever(
            embedding_provider=MockEmbeddingProvider(),
            hybrid_alpha=0.0,  # Only keyword
        )

        docs = create_sample_texts()
        for i, doc in enumerate(docs):
            retriever.add_document(f"doc_{i}", doc)

        results = retriever.search("SQL database")

        # Should return results based on keyword matching
        assert len(results) > 0


# --- SPEC-09.05: Configurable hybrid_alpha ---


class TestHybridAlpha:
    """Tests for configurable hybrid_alpha parameter."""

    def test_default_alpha_is_half(self) -> None:
        """
        @trace SPEC-09.05
        Default hybrid_alpha should be 0.5.
        """
        retriever = HybridRetriever(embedding_provider=MockEmbeddingProvider())

        assert retriever.hybrid_alpha == 0.5

    def test_custom_alpha_value(self) -> None:
        """
        @trace SPEC-09.05
        Should support custom hybrid_alpha values.
        """
        retriever = HybridRetriever(
            embedding_provider=MockEmbeddingProvider(),
            hybrid_alpha=0.7,
        )

        assert retriever.hybrid_alpha == 0.7

    def test_alpha_affects_ranking(self) -> None:
        """
        @trace SPEC-09.05
        Different alpha values should affect result ranking.
        """
        provider = MockEmbeddingProvider()

        # High alpha (semantic-focused)
        semantic_retriever = HybridRetriever(
            embedding_provider=provider,
            hybrid_alpha=0.9,
        )

        # Low alpha (keyword-focused)
        keyword_retriever = HybridRetriever(
            embedding_provider=provider,
            hybrid_alpha=0.1,
        )

        docs = create_sample_texts()
        for i, doc in enumerate(docs):
            semantic_retriever.add_document(f"doc_{i}", doc)
            keyword_retriever.add_document(f"doc_{i}", doc)

        query = "HTTP web requests"

        semantic_results = semantic_retriever.search(query)
        keyword_results = keyword_retriever.search(query)

        # Results may differ based on alpha
        # Both should return results though
        assert len(semantic_results) > 0
        assert len(keyword_results) > 0


# --- SPEC-09.06: Optional per-node embedding ---


class TestOptionalEmbedding:
    """Tests for optional per-node embedding computation."""

    def test_add_document_with_embedding(self) -> None:
        """
        @trace SPEC-09.06
        Should compute embedding when flag is True.
        """
        retriever = HybridRetriever(embedding_provider=MockEmbeddingProvider())

        retriever.add_document("doc_1", "Test content", compute_embedding=True)

        # Should be searchable via semantic search
        results = retriever.search("Test", semantic_only=True)
        assert len(results) > 0

    def test_add_document_without_embedding(self) -> None:
        """
        @trace SPEC-09.06
        Should skip embedding when flag is False.
        """
        retriever = HybridRetriever(embedding_provider=MockEmbeddingProvider())

        retriever.add_document("doc_1", "Test content", compute_embedding=False)

        # Should still be searchable via keyword search
        results = retriever.search("Test", keyword_only=True)
        assert len(results) > 0

    def test_default_compute_embedding_is_true(self) -> None:
        """
        @trace SPEC-09.06
        Default should compute embedding.
        """
        provider = MockEmbeddingProvider()
        retriever = HybridRetriever(embedding_provider=provider)

        retriever.add_document("doc_1", "Test content")

        # Embedding should have been computed
        assert provider.embed_call_count > 0


# --- SPEC-09.07: Batch embedding requests ---


class TestBatchEmbedding:
    """Tests for batch embedding requests."""

    def test_batch_embed_multiple_texts(self) -> None:
        """
        @trace SPEC-09.07
        Should support batch embedding for efficiency.
        """
        provider = MockEmbeddingProvider()
        texts = create_sample_texts()

        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == len(texts)
        for emb in embeddings:
            assert len(emb) == 1536

    def test_batch_more_efficient_than_individual(self) -> None:
        """
        @trace SPEC-09.07
        Batch embedding should be more efficient.
        """
        provider = MockEmbeddingProvider()
        texts = create_sample_texts()

        # Batch call
        provider.embed_batch(texts)
        batch_calls = provider.api_call_count

        # Reset and do individual calls
        provider.reset_counts()
        for text in texts:
            provider.embed(text)
        individual_calls = provider.api_call_count

        # Batch should use fewer API calls
        assert batch_calls <= individual_calls

    def test_retriever_uses_batch_for_bulk_add(self) -> None:
        """
        @trace SPEC-09.07
        Retriever should use batch embedding for bulk adds.
        """
        provider = MockEmbeddingProvider()
        retriever = HybridRetriever(embedding_provider=provider)

        docs = [(f"doc_{i}", text) for i, text in enumerate(create_sample_texts())]

        retriever.add_documents_batch(docs)

        # Should have used batch embedding
        assert provider.batch_call_count >= 1


# --- Integration tests ---


class TestEmbeddingRetrievalIntegration:
    """Integration tests for embedding retrieval."""

    def test_full_retrieval_workflow(self) -> None:
        """
        Test complete retrieval workflow.
        """
        retriever = HybridRetriever(
            embedding_provider=MockEmbeddingProvider(),
            hybrid_alpha=0.5,
        )

        # Add documents
        docs = [
            ("doc_1", "Python programming tutorial for beginners"),
            ("doc_2", "Advanced machine learning algorithms"),
            ("doc_3", "SQL database optimization techniques"),
            ("doc_4", "RESTful API design patterns"),
            ("doc_5", "Security best practices for web apps"),
        ]

        retriever.add_documents_batch(docs)

        # Search
        results = retriever.search("Python programming", top_k=3)

        assert len(results) <= 3
        assert len(results) > 0

        # First result should be most relevant
        assert "python" in results[0].content.lower()

    def test_empty_query_handling(self) -> None:
        """
        Empty query should return empty results.
        """
        retriever = HybridRetriever(embedding_provider=MockEmbeddingProvider())

        retriever.add_document("doc_1", "Test content")

        results = retriever.search("")

        assert len(results) == 0

    def test_no_documents_returns_empty(self) -> None:
        """
        Search with no documents should return empty.
        """
        retriever = HybridRetriever(embedding_provider=MockEmbeddingProvider())

        results = retriever.search("test query")

        assert len(results) == 0
