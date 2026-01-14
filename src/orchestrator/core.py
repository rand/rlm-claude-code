"""
Core RLM orchestration loop.

Implements: SPEC-12.02

Contains:
- Base RLMOrchestrator class
- Turn processing loop
- Event emission
- Auto-memory integration
- Error recovery strategies (SPEC-12.10)
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..api_client import ClaudeClient, Provider, init_client
from ..memory_store import MemoryStore


# ============================================================================
# Error Recovery (SPEC-12.10)
# ============================================================================


class ErrorRecoveryStrategy(Enum):
    """Strategies for recovering from errors during RLM execution."""

    RETRY_SIMPLIFIED = "retry_simplified"  # Ask LLM to simplify failed code
    FALLBACK_DIRECT = "fallback_direct"  # Fall back to direct response
    CHECKPOINT_RECOVERY = "checkpoint"  # Retry from last good checkpoint
    GRACEFUL_DEGRADATION = "graceful"  # Return partial results


@dataclass
class ErrorRecoveryConfig:
    """Configuration for error recovery behavior."""

    enabled: bool = True
    max_retries: int = 2
    strategies: list[ErrorRecoveryStrategy] = field(
        default_factory=lambda: [
            ErrorRecoveryStrategy.RETRY_SIMPLIFIED,
            ErrorRecoveryStrategy.GRACEFUL_DEGRADATION,
        ]
    )
    checkpoint_interval: int = 3  # Save checkpoint every N successful turns


@dataclass
class Checkpoint:
    """Snapshot of orchestration state for recovery."""

    turn: int
    messages: list[dict[str, str]]
    partial_results: list[str]
    repl_state: dict[str, Any]


# ============================================================================
# Core Imports
# ============================================================================

from ..complexity_classifier import should_activate_rlm
from ..config import RLMConfig, default_config
from ..context_manager import externalize_context
from ..cost_tracker import CostComponent, get_cost_tracker
from ..prompts import build_rlm_system_prompt
from ..recursive_handler import RecursiveREPL
from ..repl_environment import RLMEnvironment
from ..response_parser import ResponseAction, ResponseParser
from ..smart_router import SmartRouter
from ..trajectory import (
    StreamingTrajectory,
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryRenderer,
)
from ..types import DeferredOperation, SessionContext

if TYPE_CHECKING:
    pass


@dataclass
class OrchestrationState:
    """State for the orchestration loop."""

    depth: int = 0
    turn: int = 0
    max_turns: int = 20
    messages: list[dict[str, str]] = field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None
    # Error recovery tracking
    retry_count: int = 0
    checkpoints: list[Checkpoint] = field(default_factory=list)
    partial_results: list[str] = field(default_factory=list)
    recovered_from_error: bool = False


class RLMOrchestrator:
    """
    Main RLM orchestration loop.

    Implements: Spec §2.1 High-Level Design
    """

    def __init__(
        self,
        config: RLMConfig | None = None,
        client: ClaudeClient | None = None,
        smart_routing: bool = True,
        memory_store: MemoryStore | None = None,
        auto_memory: bool = True,
        error_recovery: ErrorRecoveryConfig | None = None,
    ):
        """
        Initialize orchestrator.

        Args:
            config: RLM configuration (uses default if None)
            client: Claude API client (creates one if None)
            smart_routing: Enable intelligent model routing based on query type
            memory_store: Optional memory store for auto-memory integration
            auto_memory: If True and memory_store provided, auto-store findings
            error_recovery: Configuration for error recovery strategies
        """
        self.config = config or default_config
        self.client = client
        self.activation_reason: str = ""
        self.parser = ResponseParser()
        self.smart_routing = smart_routing
        self._router: SmartRouter | None = None
        self._memory_store = memory_store
        self._auto_memory = auto_memory
        self._error_recovery = error_recovery or ErrorRecoveryConfig()

    def _ensure_client(self) -> ClaudeClient:
        """Ensure we have an API client."""
        if self.client is None:
            self.client = init_client(default_model=self.config.models.root_model)
        return self.client

    def _get_router(self, client: ClaudeClient) -> SmartRouter:
        """Get or create smart router."""
        if self._router is None:
            # Determine available providers from client
            available = []
            if hasattr(client, "_clients"):
                available = list(client._clients.keys())
            else:
                available = [Provider.ANTHROPIC]
            self._router = SmartRouter(available_providers=available)
        return self._router

    async def run(
        self, query: str, context: SessionContext
    ) -> AsyncIterator[TrajectoryEvent | str]:
        """
        Run RLM loop on a query.

        Implements: Spec §2 Architecture Overview

        Args:
            query: User query
            context: Session context

        Yields:
            TrajectoryEvents and final response
        """
        # Check if RLM should activate
        should_activate, self.activation_reason = should_activate_rlm(query, context)

        if not should_activate:
            # Bypass RLM, return direct
            yield TrajectoryEvent(
                type=TrajectoryEventType.FINAL,
                depth=0,
                content=f"[Direct mode: {self.activation_reason}]",
            )
            return

        # Initialize components
        client = self._ensure_client()
        renderer = TrajectoryRenderer(
            verbosity=self.config.trajectory.verbosity,
            colors=self.config.trajectory.colors,
        )
        trajectory = StreamingTrajectory(renderer)

        # Smart routing: select optimal model for this query
        selected_model = None
        routing_reason = ""
        if self.smart_routing:
            router = self._get_router(client)
            routing_decision = router.route(query)
            selected_model = routing_decision.primary_model
            routing_reason = f"{routing_decision.query_type.value} → {selected_model} ({routing_decision.reason})"

        # Initialize state
        state = OrchestrationState(
            max_turns=self.config.depth.max * 10,  # Allow multiple turns per depth
        )

        # Start event
        start_content = f"depth=0/{self.config.depth.max} • task: {self.activation_reason}"
        if routing_reason:
            start_content += f" • routing: {routing_reason}"
        start_event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            depth=0,
            content=start_content,
            metadata={
                "query": query,
                "context_tokens": context.total_tokens,
                "model": selected_model,
                "routing": routing_reason,
            },
        )
        await trajectory.emit(start_event)
        yield start_event

        # Initialize RecursiveREPL for depth management and cost tracking
        recursive_handler = RecursiveREPL(
            context=context,
            depth=0,
            config=self.config,
            trajectory=trajectory,
        )

        # Initialize REPL environment with recursive handler
        repl = RLMEnvironment(context, recursive_handler=recursive_handler)

        # Enable memory integration if available
        if self._memory_store is not None:
            repl.enable_memory(self._memory_store)

        # Analyze context
        externalized = externalize_context(context)
        analyze_event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            depth=0,
            content=f"Context: {context.total_tokens} tokens, {len(context.files)} files",
            metadata=externalized.get("context_stats"),
        )
        await trajectory.emit(analyze_event)
        yield analyze_event

        # Build system prompt
        system_prompt = build_rlm_system_prompt(context, query)

        # Initial user message
        state.messages = [{"role": "user", "content": query}]

        # Main orchestration loop
        while state.turn < state.max_turns and state.final_answer is None:
            state.turn += 1

            # Get response from Claude (using smart-routed model if enabled)
            try:
                response = await client.complete(
                    messages=state.messages,
                    system=system_prompt,
                    model=selected_model,  # Uses smart-routed model
                    max_tokens=4096,
                    component=CostComponent.ROOT_PROMPT,
                )
            except Exception as e:
                error_event = TrajectoryEvent(
                    type=TrajectoryEventType.ERROR,
                    depth=state.depth,
                    content=f"API error: {e}",
                )
                await trajectory.emit(error_event)
                yield error_event
                state.error = str(e)
                break

            # Emit reasoning event
            reason_event = TrajectoryEvent(
                type=TrajectoryEventType.REASON,
                depth=state.depth,
                content=response.content[:500] + ("..." if len(response.content) > 500 else ""),
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "model": response.model,
                    "provider": response.provider.value,
                },
            )
            await trajectory.emit(reason_event)
            yield reason_event

            # Parse response
            parsed_items = self.parser.parse(response.content)

            if not parsed_items:
                # No actionable content, might be stuck
                state.messages.append({"role": "assistant", "content": response.content})
                state.messages.append(
                    {
                        "role": "user",
                        "content": "Please continue. Use ```python blocks for REPL code, "
                        "or output FINAL: <answer> when you have the answer.",
                    }
                )
                continue

            # Process each parsed item
            for item in parsed_items:
                if item.action == ResponseAction.FINAL_ANSWER:
                    state.final_answer = item.content
                    break

                elif item.action == ResponseAction.FINAL_VAR:
                    # Get answer from REPL variable
                    var_name = item.content
                    try:
                        var_value = repl.get_variable(var_name)
                        state.final_answer = str(var_value)
                    except KeyError:
                        state.messages.append({"role": "assistant", "content": response.content})
                        state.messages.append(
                            {
                                "role": "user",
                                "content": f"Variable '{var_name}' not found. Available variables: "
                                f"{list(repl.globals.get('working_memory', {}).keys())}",
                            }
                        )
                    break

                elif item.action == ResponseAction.REPL_EXECUTE:
                    # Execute code in REPL
                    code = item.content

                    # Emit REPL exec event
                    exec_event = TrajectoryEvent(
                        type=TrajectoryEventType.REPL_EXEC,
                        depth=state.depth,
                        content=code[:200] + ("..." if len(code) > 200 else ""),
                    )
                    await trajectory.emit(exec_event)
                    yield exec_event

                    # Execute code synchronously - async ops become DeferredOperations
                    exec_result = repl.execute(code)
                    repl_result = (
                        exec_result.output if exec_result.success else f"Error: {exec_result.error}"
                    )

                    # Process any deferred async operations
                    if repl.has_pending_operations():
                        async for event in self._process_deferred_operations(
                            repl, client, state.depth, trajectory, recursive_handler
                        ):
                            yield event

                        # Re-run code now that results are available in working_memory
                        # Or format output to show deferred results
                        deferred_results = []
                        ops, batches = repl.get_pending_operations()
                        for op in ops:
                            if op.resolved:
                                deferred_results.append(
                                    f"{op.operation_id}: {str(op.result)[:200]}"
                                )
                        for batch in batches:
                            if batch.resolved:
                                deferred_results.append(
                                    f"{batch.batch_id}: {len(batch.results)} results"
                                )

                        if deferred_results:
                            repl_result = f"{repl_result}\n\nAsync results:\n" + "\n".join(
                                deferred_results
                            )

                        repl.clear_pending_operations()

                    # Emit REPL result event
                    result_event = TrajectoryEvent(
                        type=TrajectoryEventType.REPL_RESULT,
                        depth=state.depth,
                        content=str(repl_result)[:500],
                    )
                    await trajectory.emit(result_event)
                    yield result_event

                    # Add to conversation
                    state.messages.append({"role": "assistant", "content": response.content})
                    state.messages.append(
                        {
                            "role": "user",
                            "content": f"REPL output:\n```\n{repl_result}\n```\n\nContinue your analysis or provide FINAL: <answer>",
                        }
                    )

                elif item.action == ResponseAction.THINKING:
                    # Just thinking, add to messages and continue
                    state.messages.append({"role": "assistant", "content": response.content})
                    state.messages.append(
                        {
                            "role": "user",
                            "content": "Please continue with REPL actions or provide your final answer.",
                        }
                    )

            # Break if we have answer
            if state.final_answer:
                break

        # Final event
        if state.final_answer:
            final_content = state.final_answer
        elif state.error:
            final_content = f"[Error: {state.error}]"
        else:
            final_content = "[Max turns reached without answer]"

        # Emit cost report event
        cost_tracker = get_cost_tracker()
        cost_report = cost_tracker.format_report()
        cost_metadata = cost_tracker.get_summary()
        if recursive_handler:
            cost_metadata["recursive_calls"] = recursive_handler.get_cost_summary()

        cost_event = TrajectoryEvent(
            type=TrajectoryEventType.COST_REPORT,
            depth=0,
            content=cost_report,
            metadata=cost_metadata,
        )
        await trajectory.emit(cost_event)
        yield cost_event

        # Auto-memory integration: store findings and execution experience
        memory_metadata: dict[str, Any] = {}
        if self._auto_memory and self._memory_store is not None and state.final_answer:
            memory_metadata = await self._post_execution_memory_update(
                query=query,
                final_answer=state.final_answer,
                trajectory=trajectory,
                cost_metadata=cost_metadata,
                success=state.error is None,
            )

        final_event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            depth=0,
            content=final_content,
            metadata={
                "turns": state.turn,
                "cost": cost_metadata,
                "memory": memory_metadata,
            },
        )
        await trajectory.emit(final_event)
        yield final_event

        # Export trajectory if enabled
        if self.config.trajectory.export_enabled:
            import os
            import time
            from pathlib import Path

            export_dir = Path(os.path.expanduser(self.config.trajectory.export_path))
            export_dir.mkdir(parents=True, exist_ok=True)
            filename = f"trajectory_{int(time.time())}.json"
            trajectory.export_json(str(export_dir / filename))

    async def _process_deferred_operations(
        self,
        repl: RLMEnvironment,
        client: ClaudeClient,
        depth: int,
        trajectory: StreamingTrajectory,
        recursive_handler: RecursiveREPL | None = None,
    ) -> AsyncIterator[TrajectoryEvent]:
        """
        Process deferred async operations from REPL execution.

        Implements: Spec §4.2 Recursive Call Implementation (async/sync bridge)

        This handles:
        - Individual recursive_query/summarize operations
        - Parallel batch operations (llm_batch)

        All operations are executed in parallel with bounded concurrency (Semaphore(5))
        to prevent API overload while maximizing throughput.

        Uses RecursiveREPL when available for proper depth management and cost tracking.
        """
        ops, batches = repl.get_pending_operations()

        # Collect all operations for parallel execution
        all_ops: list[DeferredOperation] = list(ops)
        for batch in batches:
            all_ops.extend(batch.operations)

        if not all_ops:
            return

        # Emit start event for parallel processing
        total_ops = len(all_ops)
        start_event = TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_START,
            depth=depth + 1,
            content=f"Parallel execution: {total_ops} operations (max 5 concurrent)",
            metadata={
                "individual_ops": len(ops),
                "batch_ops": sum(len(b.operations) for b in batches),
                "batch_count": len(batches),
            },
        )
        await trajectory.emit(start_event)
        yield start_event

        # Bounded parallel execution with Semaphore(5)
        semaphore = asyncio.Semaphore(5)

        async def execute_op_bounded(op: DeferredOperation) -> tuple[str, str]:
            """Execute operation with bounded concurrency."""
            async with semaphore:
                try:
                    if recursive_handler:
                        result = await recursive_handler.recursive_query(
                            query=op.query,
                            context=op.context,
                            spawn_repl=op.spawn_repl,
                        )
                    else:
                        result = await client.recursive_query(op.query, op.context)
                    return op.operation_id, result
                except Exception as e:
                    return op.operation_id, f"[Error: {e}]"

        # Execute all operations in parallel with bounded concurrency
        results = await asyncio.gather(*[execute_op_bounded(op) for op in all_ops])

        # Build result map
        result_map: dict[str, str] = dict(results)

        # Resolve individual operations
        for op in ops:
            result = result_map.get(op.operation_id, "[Missing result]")
            repl.resolve_operation(op.operation_id, result)

        # Resolve batches
        for batch in batches:
            batch_results = [
                result_map.get(op.operation_id, "[Missing result]") for op in batch.operations
            ]
            repl.resolve_batch(batch.batch_id, batch_results)

        # Emit completion event with cost summary
        metadata: dict[str, Any] = {
            "total_operations": total_ops,
            "completed": len(results),
        }
        if recursive_handler:
            metadata["cost_summary"] = recursive_handler.get_cost_summary()

        end_event = TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_END,
            depth=depth + 1,
            content=f"Parallel execution complete: {len(results)}/{total_ops} operations",
            metadata=metadata,
        )
        await trajectory.emit(end_event)
        yield end_event

    async def _post_execution_memory_update(
        self,
        query: str,
        final_answer: str,
        trajectory: StreamingTrajectory,
        cost_metadata: dict[str, Any],
        success: bool,
    ) -> dict[str, Any]:
        """
        Store execution findings and experience in memory.

        Implements: SPEC-12.08 Auto-Memory Integration

        After RLM execution completes:
        1. Extract key findings from the final answer
        2. Store as facts with lower confidence (0.6) than explicit adds
        3. Store execution experience for strategy learning

        Args:
            query: Original user query
            final_answer: Final answer from RLM execution
            trajectory: Execution trajectory for context
            cost_metadata: Cost information for experience tracking
            success: Whether execution completed without errors

        Returns:
            Metadata about what was stored in memory
        """
        if self._memory_store is None:
            return {}

        stored_facts: list[str] = []
        experience_id: str | None = None

        # Extract facts from final answer
        facts = self._extract_facts_from_answer(final_answer)
        for fact in facts[:5]:  # Limit to 5 auto-extracted facts
            try:
                node_id = self._memory_store.create_node(
                    node_type="fact",
                    content=fact,
                    tier="task",
                    confidence=0.6,  # Lower confidence for auto-extracted
                    metadata={
                        "source": "auto_extract",
                        "query": query[:200],
                    },
                )
                stored_facts.append(node_id)
            except Exception:
                pass  # Don't fail execution on memory errors

        # Store execution experience for strategy learning
        try:
            experience_data = {
                "query": query[:500],
                "success": success,
                "total_cost": cost_metadata.get("total_cost", 0.0),
                "total_tokens": cost_metadata.get("total_tokens", 0),
                "activation_reason": self.activation_reason,
            }
            experience_id = self._memory_store.create_node(
                node_type="experience",
                content=json.dumps(experience_data),
                tier="session",
                confidence=0.8 if success else 0.4,
                metadata={
                    "outcome": "success" if success else "error",
                    "source": "rlm_execution",
                },
            )
        except Exception:
            pass  # Don't fail execution on memory errors

        return {
            "facts_stored": len(stored_facts),
            "fact_ids": stored_facts,
            "experience_id": experience_id,
        }

    def _extract_facts_from_answer(self, answer: str) -> list[str]:
        """
        Extract factual statements from an answer.

        Uses simple heuristics to identify fact-like sentences:
        - Declarative statements (not questions)
        - Contains specific identifiers (function names, file paths)
        - Reasonably short (under 200 chars)

        Args:
            answer: Final answer text

        Returns:
            List of extracted fact strings
        """
        facts: list[str] = []

        # Split into sentences
        sentences = re.split(r"[.!?\n]", answer)

        for sentence in sentences:
            sentence = sentence.strip()

            # Skip empty or too short/long
            if len(sentence) < 20 or len(sentence) > 200:
                continue

            # Skip questions
            if "?" in sentence:
                continue

            # Skip meta-statements about the analysis
            meta_patterns = [
                r"^(I |Let me|Here's|This is|Looking at|Based on)",
                r"^(To summarize|In summary|In conclusion)",
                r"(you should|you can|you need)",
            ]
            if any(re.search(p, sentence, re.IGNORECASE) for p in meta_patterns):
                continue

            # Prefer sentences with specific identifiers
            has_identifier = bool(
                re.search(r"[`\'\"][\w_]+[`\'\"]", sentence)  # Quoted identifiers
                or re.search(r"\b\w+\.(py|ts|js|go|rs)\b", sentence)  # File names
                or re.search(r"\b(function|class|method|module)\s+\w+", sentence, re.IGNORECASE)
            )

            if has_identifier:
                facts.append(sentence)
            elif len(facts) < 2:
                # Take a few general statements if we don't have many specific ones
                facts.append(sentence)

        return facts[:10]  # Cap at 10 candidates

    # =========================================================================
    # Error Recovery Methods (SPEC-12.10)
    # =========================================================================

    def _save_checkpoint(
        self,
        state: OrchestrationState,
        repl: RLMEnvironment,
    ) -> Checkpoint:
        """
        Save a checkpoint of the current orchestration state.

        Args:
            state: Current orchestration state
            repl: REPL environment

        Returns:
            Checkpoint containing state snapshot
        """
        checkpoint = Checkpoint(
            turn=state.turn,
            messages=copy.deepcopy(state.messages),
            partial_results=list(state.partial_results),
            repl_state=copy.deepcopy(repl.get_state()),
        )
        state.checkpoints.append(checkpoint)

        # Keep only last 3 checkpoints to limit memory
        if len(state.checkpoints) > 3:
            state.checkpoints = state.checkpoints[-3:]

        return checkpoint

    def _restore_checkpoint(
        self,
        state: OrchestrationState,
        repl: RLMEnvironment,
        checkpoint: Checkpoint,
    ) -> None:
        """
        Restore state from a checkpoint.

        Args:
            state: Current orchestration state to restore into
            repl: REPL environment to restore
            checkpoint: Checkpoint to restore from
        """
        state.turn = checkpoint.turn
        state.messages = copy.deepcopy(checkpoint.messages)
        state.partial_results = list(checkpoint.partial_results)
        repl.restore_state(checkpoint.repl_state)
        state.recovered_from_error = True

    async def _attempt_error_recovery(
        self,
        error: Exception,
        state: OrchestrationState,
        repl: RLMEnvironment,
        client: ClaudeClient,
        trajectory: StreamingTrajectory,
        failed_code: str | None = None,
    ) -> AsyncIterator[TrajectoryEvent]:
        """
        Attempt to recover from an error using configured strategies.

        Implements: SPEC-12.10 Error Recovery Strategies

        Tries each configured strategy in order:
        1. RETRY_SIMPLIFIED: Ask LLM to simplify failed code
        2. CHECKPOINT_RECOVERY: Restore from last good checkpoint
        3. GRACEFUL_DEGRADATION: Return partial results
        4. FALLBACK_DIRECT: Fall back to direct (non-RLM) response

        Args:
            error: The exception that occurred
            state: Current orchestration state
            repl: REPL environment
            client: API client for LLM calls
            trajectory: Trajectory for event emission
            failed_code: The code that failed (for RETRY_SIMPLIFIED)

        Yields:
            TrajectoryEvents describing recovery attempts
        """
        if not self._error_recovery.enabled:
            return

        if state.retry_count >= self._error_recovery.max_retries:
            event = TrajectoryEvent(
                type=TrajectoryEventType.ERROR,
                depth=state.depth,
                content=f"Max retries ({self._error_recovery.max_retries}) exceeded",
                metadata={"error": str(error), "recovery": "exhausted"},
            )
            await trajectory.emit(event)
            yield event
            return

        state.retry_count += 1
        error_str = str(error)

        for strategy in self._error_recovery.strategies:
            event = TrajectoryEvent(
                type=TrajectoryEventType.REASON,
                depth=state.depth,
                content=f"Attempting recovery: {strategy.value}",
                metadata={"strategy": strategy.value, "retry": state.retry_count},
            )
            await trajectory.emit(event)
            yield event

            if strategy == ErrorRecoveryStrategy.RETRY_SIMPLIFIED and failed_code:
                # Ask LLM to simplify the failed code
                async for ev in self._recovery_retry_simplified(
                    failed_code, error_str, state, client, trajectory
                ):
                    yield ev
                if state.error is None:  # Recovery succeeded
                    return

            elif strategy == ErrorRecoveryStrategy.CHECKPOINT_RECOVERY:
                # Restore from last checkpoint
                if state.checkpoints:
                    checkpoint = state.checkpoints[-1]
                    self._restore_checkpoint(state, repl, checkpoint)
                    event = TrajectoryEvent(
                        type=TrajectoryEventType.REASON,
                        depth=state.depth,
                        content=f"Restored from checkpoint at turn {checkpoint.turn}",
                        metadata={"checkpoint_turn": checkpoint.turn},
                    )
                    await trajectory.emit(event)
                    yield event
                    state.error = None  # Clear error, retry from checkpoint
                    return

            elif strategy == ErrorRecoveryStrategy.GRACEFUL_DEGRADATION:
                # Return partial results with warning
                if state.partial_results:
                    partial_answer = (
                        "[Partial results due to error]\n\n"
                        + "\n".join(state.partial_results)
                        + f"\n\n[Error: {error_str[:200]}]"
                    )
                    state.final_answer = partial_answer
                    state.error = None  # Clear error, using partial results
                    event = TrajectoryEvent(
                        type=TrajectoryEventType.FINAL,
                        depth=state.depth,
                        content="Returning partial results (graceful degradation)",
                        metadata={"partial_count": len(state.partial_results)},
                    )
                    await trajectory.emit(event)
                    yield event
                    return

            elif strategy == ErrorRecoveryStrategy.FALLBACK_DIRECT:
                # Fall back to direct response mode
                state.final_answer = (
                    "[RLM encountered an error, falling back to direct response]\n\n"
                    f"Error: {error_str[:200]}\n\n"
                    "Please try asking your question again, or rephrase it."
                )
                state.error = None
                event = TrajectoryEvent(
                    type=TrajectoryEventType.FINAL,
                    depth=state.depth,
                    content="Fallback to direct response",
                    metadata={"reason": "error_recovery"},
                )
                await trajectory.emit(event)
                yield event
                return

    async def _recovery_retry_simplified(
        self,
        failed_code: str,
        error: str,
        state: OrchestrationState,
        client: ClaudeClient,
        trajectory: StreamingTrajectory,
    ) -> AsyncIterator[TrajectoryEvent]:
        """
        Attempt recovery by asking LLM to simplify the failed code.

        Args:
            failed_code: The code that failed
            error: The error message
            state: Current orchestration state
            client: API client
            trajectory: Trajectory for events

        Yields:
            TrajectoryEvents for the simplification attempt
        """
        simplify_prompt = f"""The following Python code failed with an error:

```python
{failed_code[:1000]}
```

Error: {error[:500]}

Please provide a simpler version of this code that:
1. Avoids the error
2. Still accomplishes the same goal
3. Uses only basic operations

Respond with just the simplified code in a ```python block."""

        try:
            response = await client.complete(
                messages=[{"role": "user", "content": simplify_prompt}],
                system="You are a helpful assistant that simplifies code to avoid errors.",
                max_tokens=1024,
                component=CostComponent.ROOT_PROMPT,
            )

            event = TrajectoryEvent(
                type=TrajectoryEventType.REASON,
                depth=state.depth,
                content=f"LLM suggested simplified code ({len(response.content)} chars)",
                metadata={"simplification": "received"},
            )
            await trajectory.emit(event)
            yield event

            # Add the simplified code suggestion to messages for next iteration
            state.messages.append(
                {
                    "role": "user",
                    "content": f"The previous code failed. Here's a simplified approach:\n\n{response.content}\n\nPlease try this approach or modify it as needed.",
                }
            )
            state.error = None  # Clear error to continue

        except Exception as e:
            event = TrajectoryEvent(
                type=TrajectoryEventType.ERROR,
                depth=state.depth,
                content=f"Simplification failed: {e}",
            )
            await trajectory.emit(event)
            yield event


__all__ = [
    "OrchestrationState",
    "RLMOrchestrator",
    "ErrorRecoveryStrategy",
    "ErrorRecoveryConfig",
    "Checkpoint",
]
