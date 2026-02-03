"""
Context compression for large intermediate results.

Implements: SPEC-08.20-08.25

Provides two-stage compression (extractive + abstractive) while
preserving key information like errors, code snippets, and facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CompressionStage(Enum):
    """Stage of compression applied."""

    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"


@dataclass
class CompressionConfig:
    """
    Configuration for context compression.

    Implements: SPEC-08.23
    """

    target_tokens: int = 2000
    auto_trigger_tokens: int = 5000
    preserve_code_blocks: bool = True
    preserve_errors: bool = True
    preserve_file_refs: bool = True


@dataclass
class CompressionResult:
    """
    Result of compression operation.

    Implements: SPEC-08.25
    """

    content: str
    compressed: bool
    input_tokens: int
    output_tokens: int
    stages_applied: list[CompressionStage] = field(default_factory=list)
    relevance_scores: list[float] | None = None
    auto_triggered: bool = False
    stage: CompressionStage | None = None

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.output_tokens == 0:
            return 0.0
        return self.input_tokens / self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "compressed": self.compressed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "compression_ratio": self.compression_ratio,
            "stages_applied": [s.value for s in self.stages_applied],
            "auto_triggered": self.auto_triggered,
        }


@dataclass
class CompressionMetrics:
    """
    Metrics for compression operations.

    Implements: SPEC-08.25
    """

    total_compressions: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    _ratios: list[float] = field(default_factory=list)

    @property
    def average_ratio(self) -> float:
        """Calculate average compression ratio."""
        if not self._ratios:
            return 0.0
        return sum(self._ratios) / len(self._ratios)

    def record_compression(self, input_tokens: int, output_tokens: int) -> None:
        """Record a compression operation."""
        self.total_compressions += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        if output_tokens > 0:
            self._ratios.append(input_tokens / output_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_compressions": self.total_compressions,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "average_ratio": self.average_ratio,
        }


class RelevanceScorer:
    """
    Score sentences by relevance for extractive compression.

    Implements: SPEC-08.21
    """

    # Keywords that indicate important content
    HIGH_RELEVANCE_PATTERNS = [
        r"\berror\b",
        r"\bexception\b",
        r"\bfailed\b",
        r"\bcritical\b",
        r"\bwarning\b",
        r"\bfinding\b",
        r"\bconclusion\b",
        r"\bresult\b",
        r"\bimportant\b",
        r"\bkey\b",
        r"\bnote\b",
        r"\btodo\b",
        r"\bfixme\b",
        r"\bbug\b",
        r"\bissue\b",
        r"\btraceback\b",
        r"\bstack\s*trace\b",
        r"\bfile\s+['\"]?[\w/\\.]+['\"]?",
        r"\bline\s+\d+\b",
        r"\breturn\b",
        r"\bdef\s+\w+",
        r"\bclass\s+\w+",
        r"```",
        r"\d+%",  # Percentages
        r"\d+\.\d+",  # Numbers with decimals
    ]

    def __init__(self) -> None:
        """Initialize relevance scorer."""
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.HIGH_RELEVANCE_PATTERNS]

    def score_sentences(
        self,
        sentences: list[str],
        context: str | None = None,
    ) -> list[float]:
        """
        Score sentences by relevance.

        Args:
            sentences: List of sentences to score
            context: Optional context for scoring

        Returns:
            List of relevance scores (0-1)
        """
        scores: list[float] = []

        for sentence in sentences:
            score = self._score_sentence(sentence, context)
            scores.append(score)

        return scores

    def _score_sentence(self, sentence: str, context: str | None = None) -> float:
        """Score a single sentence."""
        base_score = 0.3  # Base relevance

        # Check for high-relevance patterns
        matches = 0
        for pattern in self._patterns:
            if pattern.search(sentence):
                matches += 1

        # Add score based on matches (max 0.5 from patterns)
        pattern_score = min(matches * 0.1, 0.5)

        # Bonus for context relevance
        context_score = 0.0
        if context:
            context_words = set(context.lower().split())
            sentence_words = set(sentence.lower().split())
            overlap = len(context_words & sentence_words)
            context_score = min(overlap * 0.05, 0.2)

        total = base_score + pattern_score + context_score
        return min(total, 1.0)


class KeyInfoPreserver:
    """
    Extract and preserve key information from content.

    Implements: SPEC-08.22
    """

    def extract_key_info(self, content: str) -> str:
        """
        Extract key information from content.

        Implements: SPEC-08.22

        Preserves:
        - Key facts and findings
        - Error messages and stack traces
        - Code snippets and file references

        Args:
            content: Content to extract from

        Returns:
            String containing preserved key information
        """
        preserved_parts: list[str] = []

        # Extract error messages
        error_patterns = [
            r"(?:Error|Exception|Failed|CRITICAL|WARNING)[:\s].*?(?=\n|$)",
            r"Traceback.*?(?=\n\n|\Z)",
            r"^\s*File\s+['\"].*?['\"].*?$",
        ]
        for pattern in error_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            preserved_parts.extend(matches)

        # Extract code blocks
        code_blocks = re.findall(r"```[\s\S]*?```", content)
        preserved_parts.extend(code_blocks)

        # Extract file references
        file_refs = re.findall(
            r"(?:File|file|src/|/[\w/]+\.(?:py|js|ts|go|rs|java|cpp|c|h))\S*",
            content,
        )
        preserved_parts.extend(file_refs)

        # Extract findings/conclusions
        findings = re.findall(
            r"(?:FINDING|CONCLUSION|RESULT|IMPORTANT|KEY)[:\s].*?(?=\n|$)",
            content,
            re.IGNORECASE,
        )
        preserved_parts.extend(findings)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_parts: list[str] = []
        for part in preserved_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        return "\n".join(unique_parts) if unique_parts else content[:500]


class ExtractiveCompressor:
    """
    Extractive compression using relevance scoring.

    Implements: SPEC-08.21 (first stage)
    """

    def __init__(self) -> None:
        """Initialize extractive compressor."""
        self.scorer = RelevanceScorer()
        self.preserver = KeyInfoPreserver()

    def compress(
        self,
        content: str,
        target_tokens: int,
        context: str | None = None,
    ) -> CompressionResult:
        """
        Compress content by selecting most relevant sentences.

        Args:
            content: Content to compress
            target_tokens: Target token count
            context: Optional context for relevance scoring

        Returns:
            CompressionResult with selected sentences
        """
        # Split into sentences
        sentences = self._split_sentences(content)
        if not sentences:
            return CompressionResult(
                content=content,
                compressed=False,
                input_tokens=self._estimate_tokens(content),
                output_tokens=self._estimate_tokens(content),
                stages_applied=[CompressionStage.EXTRACTIVE],
            )

        # Score sentences
        scores = self.scorer.score_sentences(sentences, context)

        # Select sentences up to target tokens
        selected = self._select_sentences(sentences, scores, target_tokens)

        output = " ".join(selected)
        input_tokens = self._estimate_tokens(content)
        output_tokens = self._estimate_tokens(output)

        return CompressionResult(
            content=output,
            compressed=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stages_applied=[CompressionStage.EXTRACTIVE],
            relevance_scores=scores,
        )

    def _split_sentences(self, content: str) -> list[str]:
        """Split content into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", content)
        return [s.strip() for s in sentences if s.strip()]

    def _select_sentences(
        self,
        sentences: list[str],
        scores: list[float],
        target_tokens: int,
    ) -> list[str]:
        """Select sentences up to target token count."""
        # Sort by score while tracking original indices
        indexed = list(enumerate(zip(sentences, scores)))
        indexed.sort(key=lambda x: x[1][1], reverse=True)

        selected_indices: list[int] = []
        current_tokens = 0

        for idx, (sentence, _score) in indexed:
            tokens = self._estimate_tokens(sentence)
            if current_tokens + tokens <= target_tokens:
                selected_indices.append(idx)
                current_tokens += tokens

        # Return in original order
        selected_indices.sort()
        return [sentences[i] for i in selected_indices]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # ~4 characters per token on average
        return len(text) // 4


class AbstractiveCompressor:
    """
    Abstractive compression using LLM summarization.

    Implements: SPEC-08.21 (second stage)
    """

    def __init__(self) -> None:
        """Initialize abstractive compressor."""
        self.preserver = KeyInfoPreserver()

    def compress(
        self,
        content: str,
        target_tokens: int,
    ) -> CompressionResult:
        """
        Compress content using abstractive summarization.

        In production, this would call an LLM. For now, uses
        heuristic summarization to preserve key information.

        Args:
            content: Content to compress
            target_tokens: Target token count

        Returns:
            CompressionResult with summarized content
        """
        input_tokens = self._estimate_tokens(content)

        # Extract key information first
        key_info = self.preserver.extract_key_info(content)

        # If key info is already under target, use it
        if self._estimate_tokens(key_info) <= target_tokens:
            output = key_info
        else:
            # Truncate to target (in production, LLM would summarize)
            output = self._truncate_to_tokens(key_info, target_tokens)

        output_tokens = self._estimate_tokens(output)

        return CompressionResult(
            content=output,
            compressed=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stages_applied=[CompressionStage.ABSTRACTIVE],
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4

    def _truncate_to_tokens(self, text: str, target_tokens: int) -> str:
        """Truncate text to approximately target tokens."""
        target_chars = target_tokens * 4
        if len(text) <= target_chars:
            return text
        return text[:target_chars] + "..."


class TwoStageCompressor:
    """
    Two-stage compression: extractive then abstractive.

    Implements: SPEC-08.21
    """

    def __init__(self) -> None:
        """Initialize two-stage compressor."""
        self.extractive = ExtractiveCompressor()
        self.abstractive = AbstractiveCompressor()

    def compress(
        self,
        content: str,
        target_tokens: int,
        context: str | None = None,
    ) -> CompressionResult:
        """
        Apply two-stage compression.

        Implements: SPEC-08.21
        1. Extractive: Select key sentences using relevance scoring
        2. Abstractive: LLM summarization if still over budget

        Args:
            content: Content to compress
            target_tokens: Target token count
            context: Optional context for relevance

        Returns:
            CompressionResult with compressed content
        """
        input_tokens = self._estimate_tokens(content)
        stages_applied: list[CompressionStage] = []

        # Stage 1: Extractive compression
        extractive_result = self.extractive.compress(content, target_tokens, context)
        stages_applied.append(CompressionStage.EXTRACTIVE)

        current_content = extractive_result.content
        current_tokens = self._estimate_tokens(current_content)

        # Stage 2: Abstractive if still over target
        if current_tokens > target_tokens:
            abstractive_result = self.abstractive.compress(current_content, target_tokens)
            stages_applied.append(CompressionStage.ABSTRACTIVE)
            current_content = abstractive_result.content
            current_tokens = self._estimate_tokens(current_content)

        return CompressionResult(
            content=current_content,
            compressed=True,
            input_tokens=input_tokens,
            output_tokens=current_tokens,
            stages_applied=stages_applied,
            relevance_scores=extractive_result.relevance_scores,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


class ContextCompressor:
    """
    Main context compressor with configuration and metrics.

    Implements: SPEC-08.20
    """

    def __init__(self, config: CompressionConfig | None = None) -> None:
        """
        Initialize context compressor.

        Args:
            config: Compression configuration
        """
        self.config = config or CompressionConfig()
        self.two_stage = TwoStageCompressor()
        self.metrics = CompressionMetrics()

    def compress(
        self,
        content: str,
        context: str | None = None,
    ) -> CompressionResult:
        """
        Compress content if it exceeds target tokens.

        Implements: SPEC-08.20

        Args:
            content: Content to compress
            context: Optional context for relevance

        Returns:
            CompressionResult with compressed content
        """
        input_tokens = self._estimate_tokens(content)

        # Don't compress if under target
        if input_tokens <= self.config.target_tokens:
            return CompressionResult(
                content=content,
                compressed=False,
                input_tokens=input_tokens,
                output_tokens=input_tokens,
            )

        # Apply two-stage compression
        result = self.two_stage.compress(content, self.config.target_tokens, context)

        # Record metrics
        self.metrics.record_compression(result.input_tokens, result.output_tokens)

        return result

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


class AutoCompressor:
    """
    Automatic compression for large outputs.

    Implements: SPEC-08.24
    """

    def __init__(self, config: CompressionConfig | None = None) -> None:
        """
        Initialize auto compressor.

        Args:
            config: Compression configuration
        """
        self.config = config or CompressionConfig()
        self.compressor = ContextCompressor(config)

    def maybe_compress(
        self,
        content: str,
        source: str | None = None,
    ) -> CompressionResult:
        """
        Compress content if it exceeds auto-trigger threshold.

        Implements: SPEC-08.24

        Args:
            content: Content to potentially compress
            source: Source of content (e.g., "tool_output")

        Returns:
            CompressionResult (may be uncompressed)
        """
        input_tokens = self._estimate_tokens(content)

        # Check auto-trigger threshold
        if input_tokens <= self.config.auto_trigger_tokens:
            return CompressionResult(
                content=content,
                compressed=False,
                input_tokens=input_tokens,
                output_tokens=input_tokens,
                auto_triggered=False,
            )

        # Auto-trigger compression
        result = self.compressor.compress(content)
        result.auto_triggered = True

        return result

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


__all__ = [
    "AbstractiveCompressor",
    "AutoCompressor",
    "CompressionConfig",
    "CompressionMetrics",
    "CompressionResult",
    "CompressionStage",
    "ContextCompressor",
    "ExtractiveCompressor",
    "KeyInfoPreserver",
    "RelevanceScorer",
    "TwoStageCompressor",
]
