"""
Main RLM orchestration loop.

Implements: Spec §2 Architecture Overview
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .api_client import ClaudeClient, Provider, init_client
from .complexity_classifier import should_activate_rlm
from .config import RLMConfig, default_config
from .context_manager import externalize_context
from .cost_tracker import CostComponent, get_cost_tracker
from .epistemic import (
    ClaimExtractor,
    EvidenceAuditor,
    HallucinationReport,
    VerificationConfig,
)
from .prompts import build_rlm_system_prompt
from .recursive_handler import RecursiveREPL
from .repl_environment import RLMEnvironment
from .response_parser import ResponseAction, ResponseParser
from .smart_router import SmartRouter
from .trajectory import (
    StreamingTrajectory,
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryRenderer,
)
from .types import DeferredOperation, SessionContext


@dataclass
class OrchestrationState:
    """State for the orchestration loop."""

    depth: int = 0
    turn: int = 0
    max_turns: int = 20
    messages: list[dict[str, str]] = field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None


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
        verification_config: VerificationConfig | None = None,
    ):
        """
        Initialize orchestrator.

        Args:
            config: RLM configuration (uses default if None)
            client: Claude API client (creates one if None)
            smart_routing: Enable intelligent model routing based on query type
            verification_config: Epistemic verification config (uses default if None)
        """
        self.config = config or default_config
        self.client = client
        self.activation_reason: str = ""
        self.parser = ResponseParser()
        self.smart_routing = smart_routing
        self._router: SmartRouter | None = None
        # SPEC-16.22: Epistemic verification config (always-on by default)
        self.verification_config = verification_config or VerificationConfig()

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

    @staticmethod
    def _final_answer_from_submit(submit_result: dict[str, Any]) -> str:
        """Extract a display-ready final answer from SUBMIT result payload."""
        outputs = submit_result.get("outputs")
        if isinstance(outputs, dict):
            if "answer" in outputs:
                return str(outputs["answer"])
            return json.dumps(outputs, indent=2, default=str)
        return str(outputs)

    @staticmethod
    def _clamp_unit_score(value: float) -> float:
        """Clamp a numeric score to [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

    @classmethod
    def _extract_relevance_score(cls, payload: Any) -> float:
        """Extract a normalized relevance score from an LLM response payload."""
        if isinstance(payload, bool):
            return 1.0 if payload else 0.0
        if isinstance(payload, (int, float)):
            return cls._clamp_unit_score(float(payload))
        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                return 0.0
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return cls._extract_relevance_score(parsed)
            match = re.search(r"-?\d+(?:\.\d+)?", stripped)
            if match:
                try:
                    return cls._clamp_unit_score(float(match.group(0)))
                except ValueError:
                    return 0.0
            return 0.0
        if isinstance(payload, dict):
            for key in ("score", "relevance", "relevance_score", "support_score", "value"):
                if key in payload:
                    return cls._extract_relevance_score(payload[key])
            return 0.0
        if isinstance(payload, list):
            for item in payload:
                score = cls._extract_relevance_score(item)
                if score > 0:
                    return score
            return 0.0
        return 0.0

    @staticmethod
    def _build_reduce_query(reduce_prompt: str, map_results: list[Any]) -> str:
        """Construct a reduce query from map results and a reduce prompt."""
        rendered_results = "\n".join(f"- {result}" for result in map_results)
        if "{results}" in reduce_prompt:
            try:
                return reduce_prompt.format(results=rendered_results)
            except Exception:
                pass
        return f"{reduce_prompt}\n\nMap results:\n{rendered_results}"

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
                    submit_result = exec_result.submit_result
                    if submit_result is not None and submit_result.get("status") == "success":
                        repl_result = submit_result.get("outputs")
                        state.final_answer = self._final_answer_from_submit(submit_result)
                    else:
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

                    if state.final_answer:
                        break

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

        # SPEC-16.22: Verification checkpoint (always-on unless disabled or /simple)
        verification_report: HallucinationReport | None = None
        retry_count = 0
        max_retries = self.verification_config.max_retries
        needs_user_intervention = False  # SPEC-16.24: Track if escalation needed

        if state.final_answer and self.verification_config.enabled:
            verification_report, should_retry = await self._verify_response(
                state.final_answer, context, client, trajectory
            )
            yield TrajectoryEvent(
                type=TrajectoryEventType.VERIFICATION,
                depth=0,
                content=f"Verification complete: {verification_report.verification_rate:.0%} verified",
                metadata={"report": {"flagged_claims": verification_report.flagged_claims}},
            )

            # SPEC-16.24: Handle retry with evidence focus
            while should_retry and retry_count < max_retries:
                retry_count += 1

                # Build evidence-focused retry prompt
                flagged_texts = verification_report.flagged_claim_texts[:3]  # Limit to 3
                critical_gaps = verification_report.critical_gaps[:2]  # Top 2 critical

                # List available evidence sources
                evidence_sources = list(context.files.keys())[:5]  # Limit to 5
                if context.tool_outputs:
                    evidence_sources.extend(
                        [f"tool output: {to.tool_name}" for to in context.tool_outputs[:3]]
                    )

                retry_prompt = self._build_evidence_focused_retry_prompt(
                    flagged_texts, critical_gaps, evidence_sources
                )

                state.messages.append({"role": "assistant", "content": state.final_answer})
                state.messages.append({"role": "user", "content": retry_prompt})
                state.final_answer = None

                # Use Sonnet (critical_model) for retry - this is a critical path
                retry_model = self.verification_config.critical_model

                # Emit retry event
                yield TrajectoryEvent(
                    type=TrajectoryEventType.VERIFICATION,
                    depth=0,
                    content=f"Retry {retry_count}/{max_retries}: Re-querying with evidence focus (using {retry_model})",
                    metadata={"retry_count": retry_count, "model": retry_model},
                )

                # Re-run limited turns for retry
                retry_turns = 0
                while retry_turns < 3 and state.final_answer is None:
                    retry_turns += 1
                    try:
                        response = await client.complete(
                            messages=state.messages,
                            system=system_prompt,
                            model=retry_model,  # Use critical model for retry
                            max_tokens=4096,
                            component=CostComponent.ROOT_PROMPT,
                        )
                    except Exception as e:
                        state.error = str(e)
                        break

                    parsed_items = self.parser.parse(response.content)
                    for item in parsed_items:
                        if item.action == ResponseAction.FINAL_ANSWER:
                            state.final_answer = item.content
                            break
                    if not state.final_answer:
                        state.messages.append({"role": "assistant", "content": response.content})
                        state.messages.append(
                            {"role": "user", "content": "Please provide FINAL: <answer>"}
                        )

                # Re-verify after retry
                if state.final_answer:
                    verification_report, should_retry = await self._verify_response(
                        state.final_answer, context, client, trajectory
                    )

            # SPEC-16.24: Escalate to "ask" mode if retries exhausted and still has issues
            if should_retry and retry_count >= max_retries:
                needs_user_intervention = True
                yield TrajectoryEvent(
                    type=TrajectoryEventType.VERIFICATION,
                    depth=0,
                    content=f"Escalation: {verification_report.flagged_claims} claims still unverified after {retry_count} retries",
                    metadata={
                        "escalation": "ask",
                        "flagged_claims": verification_report.flagged_claim_texts,
                        "critical_gaps": [g.claim_text for g in verification_report.critical_gaps],
                    },
                )

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

        # Build final metadata with optional verification info
        final_metadata: dict[str, Any] = {
            "turns": state.turn,
            "cost": cost_metadata,
        }
        if verification_report:
            final_metadata["verification"] = {
                "total_claims": verification_report.total_claims,
                "verified_claims": verification_report.verified_claims,
                "flagged_claims": verification_report.flagged_claims,
                "verification_rate": verification_report.verification_rate,
                "has_critical_gaps": verification_report.has_critical_gaps,
                "retry_count": retry_count,
                "needs_user_intervention": needs_user_intervention,
            }

        final_event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            depth=0,
            content=final_content,
            metadata=final_metadata,
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
                "max_concurrent": 5,
                "max_attempts": 2,
            },
        )
        await trajectory.emit(start_event)
        yield start_event

        # Bounded parallel execution with deterministic retry policy.
        max_concurrent = 5
        max_attempts = 2
        retry_backoff_s = 0.05
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_op_bounded(op: DeferredOperation) -> tuple[str, Any, dict[str, Any]]:
            """Execute one deferred operation with bounded concurrency + retries."""
            async with semaphore:
                for attempt in range(1, max_attempts + 1):
                    try:
                        if recursive_handler:
                            result = await recursive_handler.recursive_query(
                                query=op.query,
                                context=op.context,
                                spawn_repl=op.spawn_repl,
                            )
                        else:
                            result = await client.recursive_query(op.query, op.context)
                        return op.operation_id, result, {"status": "success", "attempts": attempt}
                    except Exception as e:
                        if attempt >= max_attempts:
                            return op.operation_id, f"[Error: {e}]", {
                                "status": "error",
                                "attempts": attempt,
                                "error": str(e),
                            }
                        await asyncio.sleep(retry_backoff_s * attempt)
            raise RuntimeError(f"Unreachable: max_attempts={max_attempts}")

        # Execute all operations in parallel with bounded concurrency
        results = await asyncio.gather(*[execute_op_bounded(op) for op in all_ops])

        # Build result and telemetry maps
        op_lookup = {op.operation_id: op for op in all_ops}
        result_map: dict[str, Any] = {}
        op_telemetry: list[dict[str, Any]] = []
        for op_id, result, status in results:
            result_map[op_id] = result
            op_telemetry.append(
                {
                    "operation_id": op_id,
                    "operation_type": op_lookup[op_id].operation_type,
                    **status,
                }
            )

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

            # SPEC-01.04: Execute reduce phase for map_reduce batches.
            reduce_prompt = batch.metadata.get("reduce_prompt")
            if isinstance(reduce_prompt, str) and reduce_prompt.strip():
                reduce_op = DeferredOperation(
                    operation_id=f"{batch.batch_id}_reduce",
                    operation_type="reduce",
                    query=self._build_reduce_query(reduce_prompt, batch_results),
                    context="\n".join(str(item) for item in batch_results),
                    spawn_repl=False,
                    metadata={"batch_id": batch.batch_id},
                )
                op_lookup[reduce_op.operation_id] = reduce_op
                reduce_id, reduce_result, reduce_status = await execute_op_bounded(reduce_op)
                result_map[reduce_id] = reduce_result
                op_telemetry.append(
                    {
                        "operation_id": reduce_id,
                        "operation_type": reduce_op.operation_type,
                        **reduce_status,
                    }
                )
                batch.metadata["reduce_result"] = reduce_result
                batch.metadata["reduce_status"] = reduce_status["status"]
                batch.metadata["reduce_attempts"] = reduce_status.get("attempts", 1)
                repl.globals["working_memory"][f"{batch.batch_id}_reduced"] = reduce_result

            # SPEC-01.10: Materialize reranked results from deferred LLM scoring.
            if batch.metadata.get("batch_type") == "find_relevant_llm_scoring":
                raw_top_k = batch.metadata.get("top_k", 5)
                top_k = raw_top_k if isinstance(raw_top_k, int) else 5
                candidates = batch.metadata.get("candidates", [])
                reranked: list[tuple[str, float]] = []

                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    op_id = candidate.get("operation_id")
                    chunk = candidate.get("chunk")
                    if not isinstance(op_id, str) or not isinstance(chunk, str):
                        continue

                    raw_keyword = candidate.get("keyword_score", 0.0)
                    keyword_score = float(raw_keyword) if isinstance(raw_keyword, (int, float)) else 0.0
                    llm_score = self._extract_relevance_score(result_map.get(op_id))
                    combined = self._clamp_unit_score(0.7 * llm_score + 0.3 * keyword_score)
                    reranked.append((chunk, round(combined, 6)))

                reranked.sort(key=lambda item: item[1], reverse=True)
                reranked_results = reranked[: max(0, top_k)]
                batch.metadata["reranked_results"] = reranked_results
                batch.metadata["llm_scoring_applied"] = True
                repl.globals["working_memory"][f"{batch.batch_id}_reranked"] = reranked_results

        # Emit completion event with cost summary
        failed = sum(1 for op in op_telemetry if op["status"] == "error")
        succeeded = len(op_telemetry) - failed
        retried = sum(1 for op in op_telemetry if op.get("attempts", 1) > 1)
        metadata: dict[str, Any] = {
            "total_operations": len(op_telemetry),
            "initial_operations": total_ops,
            "completed": len(op_telemetry),
            "completed_initial": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "retried": retried,
            "max_concurrent": max_concurrent,
            "max_attempts": max_attempts,
            "operations": op_telemetry,
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

    async def _verify_response(
        self,
        response: str,
        context: SessionContext,
        client: ClaudeClient,
        trajectory: StreamingTrajectory,
    ) -> tuple[HallucinationReport, bool]:
        """
        Verify response for hallucinations via epistemic verification.

        Implements: SPEC-16.22 Verification checkpoint in orchestrator

        This checkpoint:
        1. Extracts claims from the response
        2. Audits each claim against available evidence
        3. Returns a HallucinationReport with results
        4. Determines if retry is needed based on config

        Args:
            response: The response text to verify
            context: Session context with evidence (files, tool outputs)
            client: LLM client for verification calls
            trajectory: Trajectory for event logging

        Returns:
            Tuple of (HallucinationReport, should_retry)
        """
        if not self.verification_config.enabled:
            # Return empty report if verification disabled
            return HallucinationReport(response_id="disabled"), False

        # Build evidence dict from context
        evidence: dict[str, str] = {}
        for path, content in context.files.items():
            evidence[path] = content
        for i, tool_output in enumerate(context.tool_outputs):
            evidence[f"tool_{i}_{tool_output.tool_name}"] = tool_output.content

        # Skip verification if no evidence available
        if not evidence:
            return HallucinationReport(response_id="no_evidence"), False

        # Initialize extractor and auditor
        extractor = ClaimExtractor(
            client=client,
            default_model=self.verification_config.verification_model,
            max_claims=self.verification_config.max_claims_per_response,
        )
        auditor = EvidenceAuditor(
            client=client,
            default_model=self.verification_config.verification_model,
            critical_model=self.verification_config.critical_model,
            support_threshold=self.verification_config.support_threshold,
        )

        # Extract claims
        extraction_result = await extractor.extract_claims(response)
        claims = extraction_result.claims

        # Apply sampling if in sample mode
        if self.verification_config.mode == "sample":
            sampled_claims = [
                c
                for i, c in enumerate(claims)
                if self.verification_config.should_verify_claim(i, c.is_critical)
            ]
            claims = sampled_claims
        elif self.verification_config.mode == "critical_only":
            claims = [c for c in claims if c.is_critical]

        # Map claims to evidence
        if claims:
            claims = await extractor.map_claims_to_evidence(claims, evidence)

        # Audit claims
        audit_result = await auditor.audit_claims(claims, evidence)

        # Build HallucinationReport
        report = HallucinationReport(
            response_id=extraction_result.response_id,
            total_claims=len(extraction_result.claims),
            verified_claims=audit_result.total_claims - audit_result.flagged_count,
            flagged_claims=audit_result.flagged_count,
        )

        # Add claim verifications and gaps to report
        for result in audit_result.results:
            report.add_claim(result.verification)
            for gap in result.gaps:
                report.add_gap(gap)

        # Determine if should retry based on config
        should_retry = False
        if report.has_critical_gaps and self.verification_config.on_failure == "retry":
            should_retry = True

        # Emit verification event
        verification_event = TrajectoryEvent(
            type=TrajectoryEventType.VERIFICATION,
            depth=0,
            content=f"Verification: {report.verified_claims}/{report.total_claims} claims verified"
            + (f" ({report.flagged_claims} flagged)" if report.flagged_claims > 0 else ""),
            metadata={
                "total_claims": report.total_claims,
                "verified_claims": report.verified_claims,
                "flagged_claims": report.flagged_claims,
                "has_critical_gaps": report.has_critical_gaps,
                "overall_confidence": report.overall_confidence,
                "should_retry": should_retry,
            },
        )
        await trajectory.emit(verification_event)

        return report, should_retry

    def _build_evidence_focused_retry_prompt(
        self,
        flagged_texts: list[str],
        critical_gaps: list,
        evidence_sources: list[str],
    ) -> str:
        """
        Build an evidence-focused retry prompt.

        Implements: SPEC-16.24 Retry with evidence focus

        Creates a prompt that:
        1. Lists the specific claims that couldn't be verified
        2. Identifies the type of issues found (phantom citations, contradictions, etc.)
        3. Lists available evidence sources to consult
        4. Requests a grounded response with explicit citations

        Args:
            flagged_texts: List of claim texts that were flagged
            critical_gaps: List of EpistemicGap objects for critical issues
            evidence_sources: List of available evidence source names

        Returns:
            Formatted retry prompt string
        """
        prompt_parts = [
            "## Verification Failed\n",
            "Some claims in your response could not be verified against the available evidence.\n\n",
        ]

        # List flagged claims
        if flagged_texts:
            prompt_parts.append("### Unverified Claims\n")
            for claim in flagged_texts:
                prompt_parts.append(f"- {claim}\n")
            prompt_parts.append("\n")

        # Explain critical issues
        if critical_gaps:
            prompt_parts.append("### Critical Issues Found\n")
            for gap in critical_gaps:
                gap_type = getattr(gap, "gap_type", "unknown")
                if gap_type == "phantom_citation":
                    prompt_parts.append("- **Phantom citation**: Referenced source not found\n")
                elif gap_type == "contradicted":
                    prompt_parts.append("- **Contradiction**: Evidence contradicts claim\n")
                elif gap_type == "unsupported":
                    prompt_parts.append("- **Unsupported**: No evidence supports this claim\n")
                elif gap_type == "over_extrapolation":
                    prompt_parts.append("- **Over-extrapolation**: Claim goes beyond evidence\n")
            prompt_parts.append("\n")

        # List available evidence
        if evidence_sources:
            prompt_parts.append("### Available Evidence Sources\n")
            prompt_parts.append("Please ground your response in these sources:\n")
            for source in evidence_sources:
                prompt_parts.append(f"- `{source}`\n")
            prompt_parts.append("\n")

        # Instructions for grounded response
        prompt_parts.extend(
            [
                "### Requirements for Revised Response\n",
                "1. **Only make claims you can support** with the available evidence\n",
                "2. **Cite specific sources** when making factual claims (e.g., 'According to src/file.py...')\n",
                "3. **Acknowledge uncertainty** if evidence is incomplete or ambiguous\n",
                "4. **Remove or qualify** any claims that cannot be verified\n\n",
                "Provide a revised FINAL: <answer> that addresses these issues.",
            ]
        )

        return "".join(prompt_parts)


async def main():
    """CLI entry point for testing."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="RLM Orchestrator")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument(
        "--verbosity", default="normal", choices=["minimal", "normal", "verbose", "debug"]
    )
    parser.add_argument("--export-trajectory", help="Path to export trajectory JSON")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    # Check for API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: export ANTHROPIC_API_KEY=your-key-here")
        print("Or pass --api-key argument")
        return

    # Create context with actual project files for CLI testing
    files = {}
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(src_dir):
        if filename.endswith(".py"):
            filepath = os.path.join(src_dir, filename)
            try:
                with open(filepath) as f:
                    files[filename] = f.read()
            except Exception:
                pass

    context = SessionContext(files=files)

    config = RLMConfig()
    config.trajectory.verbosity = args.verbosity

    if args.export_trajectory:
        config.trajectory.export_enabled = True
        config.trajectory.export_path = args.export_trajectory

    # Initialize client
    client = init_client(api_key=api_key)

    orchestrator = RLMOrchestrator(config, client)

    async for event in orchestrator.run(args.query, context):
        if isinstance(event, TrajectoryEvent):
            renderer = TrajectoryRenderer(verbosity=args.verbosity)
            print(renderer.render_event(event))


if __name__ == "__main__":
    asyncio.run(main())
