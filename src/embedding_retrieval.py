"""
Embedding-based memory retrieval.

Implements: SPEC-09.01-09.07 Embedding-Based Memory Retrieval
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding computation.

    Implements: SPEC-09.02
    """

    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 10


@dataclass
class HybridSearchResult:
    """
    Result from hybrid search.

    Implements: SPEC-09.04
    """

    node_id: str
    content: str
    semantic_score: float
    keyword_score: float
    hybrid_alpha: float

    @property
    def hybrid_score(self) -> float:
        """Calculate hybrid score from semantic and keyword scores."""
        return (
            self.hybrid_alpha * self.semantic_score + (1 - self.hybrid_alpha) * self.keyword_score
        )


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Implements: SPEC-09.01
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Compute embedding for single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for batch of texts."""
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing.

    Produces deterministic embeddings based on text hash.
    """

    def __init__(self, dimensions: int = 1536):
        """
        Initialize mock provider.

        Args:
            dimensions: Embedding dimensions
        """
        self.dimensions = dimensions
        self.embed_call_count = 0
        self.batch_call_count = 0
        self.api_call_count = 0

    def embed(self, text: str) -> list[float]:
        """Generate deterministic embedding from text hash."""
        self.embed_call_count += 1
        self.api_call_count += 1
        return self._generate_embedding(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch."""
        self.batch_call_count += 1
        self.api_call_count += 1  # Single API call for batch
        return [self._generate_embedding(text) for text in texts]

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate deterministic normalized embedding from text."""
        # Use hash for determinism
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Generate raw values from hash
        raw = []
        for i in range(self.dimensions):
            # Use different parts of hash for different dimensions
            byte_idx = i % len(hash_bytes)
            val = (hash_bytes[byte_idx] + i) / 255.0 - 0.5
            raw.append(val)

        # Normalize to unit vector
        magnitude = math.sqrt(sum(x * x for x in raw))
        if magnitude > 0:
            return [x / magnitude for x in raw]
        return raw

    def reset_counts(self) -> None:
        """Reset call counters."""
        self.embed_call_count = 0
        self.batch_call_count = 0
        self.api_call_count = 0


class EmbeddingStore:
    """
    Storage for embeddings using SQLite.

    Implements: SPEC-09.03

    Note: Uses simple table storage. For production, would use
    vec0 virtual table or similar vector extension.
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize embedding store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS node_embeddings (
                node_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def store(self, node_id: str, embedding: list[float]) -> None:
        """
        Store embedding for node.

        Args:
            node_id: Node identifier
            embedding: Embedding vector
        """
        embedding_json = json.dumps(embedding)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO node_embeddings (node_id, embedding)
            VALUES (?, ?)
            """,
            (node_id, embedding_json),
        )
        self.conn.commit()

    def get(self, node_id: str) -> list[float] | None:
        """
        Retrieve embedding for node.

        Args:
            node_id: Node identifier

        Returns:
            Embedding vector or None if not found
        """
        cursor = self.conn.execute(
            "SELECT embedding FROM node_embeddings WHERE node_id = ?",
            (node_id,),
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def delete(self, node_id: str) -> bool:
        """Delete embedding for node."""
        cursor = self.conn.execute(
            "DELETE FROM node_embeddings WHERE node_id = ?",
            (node_id,),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


@dataclass
class Document:
    """Internal document representation."""

    node_id: str
    content: str
    embedding: list[float] | None = None


class HybridRetriever:
    """
    Hybrid retriever combining keyword and semantic search.

    Implements: SPEC-09.04-09.07

    Combines:
    - Keyword search via simple text matching
    - Semantic search via embedding similarity
    - Configurable hybrid_alpha for blending
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        hybrid_alpha: float = 0.5,
        embedding_store: EmbeddingStore | None = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            embedding_provider: Provider for computing embeddings
            hybrid_alpha: Weight for semantic vs keyword (0=keyword, 1=semantic)
            embedding_store: Optional embedding store
        """
        self.embedding_provider = embedding_provider
        self.hybrid_alpha = hybrid_alpha
        self.embedding_store = embedding_store or EmbeddingStore(":memory:")
        self._documents: dict[str, Document] = {}

    def add_document(
        self,
        node_id: str,
        content: str,
        compute_embedding: bool = True,
    ) -> None:
        """
        Add document to retriever.

        Implements: SPEC-09.06

        Args:
            node_id: Document identifier
            content: Document content
            compute_embedding: Whether to compute embedding
        """
        embedding = None
        if compute_embedding:
            embedding = self.embedding_provider.embed(content)
            self.embedding_store.store(node_id, embedding)

        self._documents[node_id] = Document(
            node_id=node_id,
            content=content,
            embedding=embedding,
        )

    def add_documents_batch(
        self,
        documents: list[tuple[str, str]],
        compute_embedding: bool = True,
    ) -> None:
        """
        Add multiple documents with batch embedding.

        Implements: SPEC-09.07

        Args:
            documents: List of (node_id, content) tuples
            compute_embedding: Whether to compute embeddings
        """
        if not documents:
            return

        if compute_embedding:
            # Batch compute embeddings
            contents = [content for _, content in documents]
            embeddings = self.embedding_provider.embed_batch(contents)

            for (node_id, content), embedding in zip(documents, embeddings):
                self.embedding_store.store(node_id, embedding)
                self._documents[node_id] = Document(
                    node_id=node_id,
                    content=content,
                    embedding=embedding,
                )
        else:
            for node_id, content in documents:
                self._documents[node_id] = Document(
                    node_id=node_id,
                    content=content,
                    embedding=None,
                )

    def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_only: bool = False,
        keyword_only: bool = False,
    ) -> list[HybridSearchResult]:
        """
        Search documents using hybrid retrieval.

        Implements: SPEC-09.04

        Args:
            query: Search query
            top_k: Maximum results to return
            semantic_only: Use only semantic search
            keyword_only: Use only keyword search

        Returns:
            List of search results sorted by hybrid score
        """
        if not query or not self._documents:
            return []

        # Compute query embedding for semantic search
        query_embedding = None
        if not keyword_only:
            query_embedding = self.embedding_provider.embed(query)

        results = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for doc in self._documents.values():
            # Keyword score
            keyword_score = 0.0
            if not semantic_only:
                keyword_score = self._compute_keyword_score(doc.content, query_terms)

            # Semantic score
            semantic_score = 0.0
            if not keyword_only and query_embedding and doc.embedding:
                semantic_score = self._cosine_similarity(query_embedding, doc.embedding)

            # Determine effective alpha
            if semantic_only:
                effective_alpha = 1.0
            elif keyword_only:
                effective_alpha = 0.0
            else:
                effective_alpha = self.hybrid_alpha

            result = HybridSearchResult(
                node_id=doc.node_id,
                content=doc.content,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                hybrid_alpha=effective_alpha,
            )

            # Only include if has some relevance
            if result.hybrid_score > 0:
                results.append(result)

        # Sort by hybrid score descending
        results.sort(key=lambda r: r.hybrid_score, reverse=True)

        return results[:top_k]

    def _compute_keyword_score(self, content: str, query_terms: set[str]) -> float:
        """Compute keyword match score."""
        content_lower = content.lower()
        content_terms = set(content_lower.split())

        if not query_terms:
            return 0.0

        # Jaccard-like overlap
        overlap = len(query_terms & content_terms)
        return overlap / len(query_terms)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)


__all__ = [
    "Document",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingStore",
    "HybridRetriever",
    "HybridSearchResult",
    "MockEmbeddingProvider",
]
