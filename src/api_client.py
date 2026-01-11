"""
Claude API client wrapper for RLM.

Implements: Spec §5 Model Integration
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

# Auto-load .env from project root
_project_root = Path(__file__).parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

from .cost_tracker import CostComponent, CostTracker, get_cost_tracker


@dataclass
class APIResponse:
    """Response from Claude API."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A chunk from streaming response."""

    text: str
    is_final: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class ClaudeClient:
    """
    Claude API client with cost tracking.

    Implements: Spec §5.1 API Integration
    """

    # Model mappings
    MODELS = {
        "opus": "claude-opus-4-5-20251101",
        "sonnet": "claude-sonnet-4-20250514",
        "haiku": "claude-haiku-4-5-20251001",
    }

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "sonnet",
        cost_tracker: CostTracker | None = None,
    ):
        """
        Initialize client.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            default_model: Default model to use
            cost_tracker: Cost tracker instance
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.default_model = self._resolve_model(default_model)
        self.cost_tracker = cost_tracker or get_cost_tracker()

    def _resolve_model(self, model: str) -> str:
        """Resolve model shorthand to full name."""
        return self.MODELS.get(model, model)

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
    ) -> APIResponse:
        """
        Get completion from Claude.

        Args:
            messages: Conversation messages
            system: System prompt
            model: Model to use (defaults to default_model)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            component: Cost component for tracking

        Returns:
            APIResponse with content and usage
        """
        model = self._resolve_model(model) if model else self.default_model

        # Build request
        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            request_params["system"] = system

        # Make API call
        response = self.client.messages.create(**request_params)

        # Extract content
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        # Track costs
        self.cost_tracker.record_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
            component=component,
        )

        return APIResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
            stop_reason=response.stop_reason,
        )

    async def complete_streaming(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
    ):
        """
        Get streaming completion from Claude.

        Args:
            messages: Conversation messages
            system: System prompt
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            component: Cost component

        Yields:
            StreamChunk objects
        """
        model = self._resolve_model(model) if model else self.default_model

        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            request_params["system"] = system

        input_tokens = 0
        output_tokens = 0

        with self.client.messages.stream(**request_params) as stream:
            for text in stream.text_stream:
                yield StreamChunk(text=text)

            # Get final message for token counts
            final_message = stream.get_final_message()
            input_tokens = final_message.usage.input_tokens
            output_tokens = final_message.usage.output_tokens

        # Track costs
        self.cost_tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            component=component,
        )

        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def recursive_query(
        self,
        query: str,
        context: str,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """
        Make a recursive sub-query.

        Implements: Spec §3.3 Recursive Call Protocol

        Args:
            query: Question to answer
            context: Context chunk to analyze
            model: Model to use (defaults to haiku for efficiency)
            max_tokens: Max response tokens

        Returns:
            Answer string
        """
        from .prompts import build_recursive_prompt

        # Use faster model for recursive calls by default
        model = model or "haiku"

        messages = [{"role": "user", "content": build_recursive_prompt(query, context)}]

        response = await self.complete(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            component=CostComponent.RECURSIVE_CALL,
        )

        return response.content

    async def summarize(
        self,
        content: str,
        max_tokens: int = 500,
        model: str | None = None,
    ) -> str:
        """
        Summarize content.

        Args:
            content: Content to summarize
            max_tokens: Target summary length
            model: Model to use

        Returns:
            Summary string
        """
        from .prompts import build_summarization_prompt

        model = model or "haiku"

        messages = [
            {"role": "user", "content": build_summarization_prompt(content, max_tokens)}
        ]

        response = await self.complete(
            messages=messages,
            model=model,
            max_tokens=max_tokens + 100,  # Buffer for formatting
            component=CostComponent.SUMMARIZATION,
        )

        return response.content


# Global client instance
_client: ClaudeClient | None = None


def get_client() -> ClaudeClient:
    """Get global Claude client."""
    global _client
    if _client is None:
        _client = ClaudeClient()
    return _client


def init_client(api_key: str | None = None, **kwargs: Any) -> ClaudeClient:
    """Initialize global client with options."""
    global _client
    _client = ClaudeClient(api_key=api_key, **kwargs)
    return _client


__all__ = [
    "APIResponse",
    "ClaudeClient",
    "StreamChunk",
    "get_client",
    "init_client",
]
