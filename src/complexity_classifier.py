"""
Task complexity classification for RLM activation.

Implements: Spec §6.3 Task Complexity-Based Activation

Delegates to rlm_core.PatternClassifier for classification.
"""

import re
from pathlib import Path

import rlm_core

from .types import MessageRole, SessionContext, TaskComplexitySignals


def _convert_to_rlm_core_context(context: SessionContext) -> "rlm_core.SessionContext":
    """
    Convert Python SessionContext to rlm_core.SessionContext.

    This adapter bridges the Python implementation types to the Rust bindings.
    """
    core_ctx = rlm_core.SessionContext()

    # Convert messages
    for msg in context.messages:
        if msg.role == MessageRole.USER:
            core_ctx.add_user_message(msg.content)
        elif msg.role == MessageRole.ASSISTANT:
            core_ctx.add_assistant_message(msg.content)

    # Convert tool outputs
    for output in context.tool_outputs:
        core_output = rlm_core.ToolOutput(output.tool_name, output.content)
        core_ctx.add_tool_output(core_output)

    # Cache files
    for file_path, content in context.files.items():
        core_ctx.cache_file(file_path, content)

    return core_ctx


def _detect_state_changes(context: SessionContext) -> bool:
    """
    Detect if recent tool outputs indicate state changes.

    State-changing operations include file edits, writes, and
    bash commands that modify the filesystem.
    """
    state_changing_tools = {"Edit", "Write", "NotebookEdit"}
    state_changing_bash_patterns = [
        r"\b(mkdir|rm|mv|cp|touch|chmod|chown)\b",
        r"\bgit\s+(add|commit|push|checkout|merge|rebase)\b",
        r"\b(npm|yarn|pip|cargo)\s+(install|remove|update)\b",
    ]

    for output in context.tool_outputs[-10:]:  # Check last 10 tool outputs
        if output.tool_name in state_changing_tools:
            return True
        if output.tool_name == "Bash":
            for pattern in state_changing_bash_patterns:
                if re.search(pattern, output.content):
                    return True
    return False


def _detect_confusion(context: SessionContext) -> bool:
    """
    Detect if the previous assistant turn showed signs of confusion.

    Confusion indicators include apologies, corrections, uncertainty,
    and explicit statements of misunderstanding.
    """
    confusion_patterns = [
        r"\b(sorry|apologi[zs]e|my (mistake|bad|error))\b",
        r"\b(actually|wait|correction|let me (re|correct))\b",
        r"\b(i (was|am) (wrong|mistaken|confused))\b",
        r"\b(not sure|uncertain|unclear)\b",
        r"\b(misunderstood|misread|overlooked)\b",
        r"\bthat('s| is) not (right|correct|what)\b",
    ]

    # Find the last assistant message
    for msg in reversed(context.messages):
        if msg.role == MessageRole.ASSISTANT:
            content_lower = msg.content.lower()
            for pattern in confusion_patterns:
                if re.search(pattern, content_lower):
                    return True
            break  # Only check the most recent assistant message
    return False


def extract_complexity_signals(prompt: str, context: SessionContext) -> TaskComplexitySignals:
    """
    Extract complexity signals from prompt and context.

    Implements: Spec §6.3 Task Complexity-Based Activation

    Must be fast (<50ms) so uses heuristics, not LLM calls.
    """
    prompt_lower = prompt.lower()

    # File reference patterns - count total file references across all patterns
    file_extension_pattern = r"\b\w+\.(ts|js|py|go|rs|tsx|jsx)\b"
    file_matches = len(re.findall(file_extension_pattern, prompt_lower))

    # Also check for module pair mentions (auth and api, etc.)
    module_pair_pattern = r"\b(auth|api|db|ui|test|config)\b.*\b(auth|api|db|ui|test|config)\b"
    has_module_pair = bool(re.search(module_pair_pattern, prompt_lower))

    # Also check for conjunction patterns suggesting multiple targets
    conjunction_pattern = r"\b(and|also|plus)\s+(update|fix|change|modify)"
    has_conjunction = bool(re.search(conjunction_pattern, prompt_lower))

    # Multiple files if: 2+ explicit files, or module pair, or conjunction with file
    references_multiple = (
        file_matches >= 2 or has_module_pair or (has_conjunction and file_matches >= 1)
    )

    # Cross-context reasoning patterns
    cross_context_patterns = [
        r"\bwhy\b.*\b(when|if|given|since)\b",
        r"\bhow\b.*\b(relate|connect|affect|impact)\b",
        r"\bwhat\b.*\b(cause|led to|result)\b",
        r"\b(trace|follow|track)\b.*\b(through|across)\b",
    ]

    # Temporal patterns
    temporal_patterns = [
        r"\b(before|after|since|when|changed|used to|previously)\b",
        r"\b(history|log|commit|version|diff)\b",
        r"\blast\s+(time|session|attempt)\b",
    ]

    # Pattern search indicators
    pattern_patterns = [
        r"\b(find|search|locate|grep|all|every|each)\b.*\b(where|that|which)\b",
        r"\bhow many\b",
        r"\blist\s+(all|every)\b",
    ]

    # Debug indicators (using word stems to catch variations like failing/failed)
    debug_patterns = [
        r"\b(error|exception|fail\w*|crash\w*|bug|issue|broken)\b",
        r"\b(stack\s*trace|traceback|stderr)\b",
        r"\b(debug\w*|diagnos\w*|investigat\w*|troubleshoot\w*)\b",
    ]

    # Exhaustive search indicators - need systematic enumeration
    # Use .* to allow words between key terms (e.g., "find and fix all")
    exhaustive_patterns = [
        r"\b(find|list|show|get)\b.*\b(all|every)\b",
        r"\b(ensure|check|verify)\b.*\b(all|every|each)\b",
        r"\b(comprehensive|exhaustive|complete)\s+(list|search|scan|review)\b",
        r"\ball\s+(the\s+)?(places|instances|usages|occurrences|references)\b",
        r"\b(fix|update|change|remove)\b.*\b(all|every|each)\b",
    ]

    # Security and review patterns - require careful multi-file analysis
    security_review_patterns = [
        r"\b(security|vulnerabilit|exploit|attack|injection|xss|csrf|auth)\w*\b",
        r"\b(review|audit)\b.*\b(code|pr|pull\s*request|changes?|implementation)\b",
        r"\b(code|pr|pull\s*request)\b.*\b(review|audit)\b",
        r"\b(check|scan|analyze)\b.*\b(security|vulnerabil|risk)\b",
    ]

    # Architecture and design patterns - require system-wide understanding
    architecture_patterns = [
        r"\b(architecture|architect)\b",
        r"\b(system|overall|high.?level)\s+(design|structure|overview)\b",
        r"\b(summarize|explain|understand)\b.*\b(codebase|project|system|architecture)\b",
        r"\b(how\s+does|how\s+do)\b.*\b(work|fit|connect|integrate|interact|communicate|call|use)\b",
        r"\b(design|structure)\s+(of|for)\s+(the|this)\b",
        r"\b(refactor|restructure|reorganize)\b",
        r"\b(explain|describe|trace)\b.*\b(data\s*flow|flow|path|pipeline)\b",
        r"\b(explain|describe|understand)\b.*\bhow\b.*\bworks?\b",
    ]

    # User intent: thorough mode signals
    thorough_patterns = [
        r"\bmake\s+sure\b",
        r"\bbe\s+careful\b",
        r"\b(thorough|thoroughly)\b",
        r"\bdon'?t\s+miss\b",
        r"\bcheck\s+everything\b",
        r"\b(verify|validate|confirm)\s+(all|every|each)\b",
        r"\b(important|critical|crucial)\b",
    ]

    # User intent: fast mode signals
    fast_patterns = [
        r"\b(quick|quickly)\b",
        r"\bjust\s+(show|tell|give)\b",
        r"\bbriefly\b",
        r"\bsimple\s+(answer|explanation)\b",
    ]

    return TaskComplexitySignals(
        references_multiple_files=references_multiple,
        requires_cross_context_reasoning=any(
            re.search(p, prompt_lower) for p in cross_context_patterns
        ),
        involves_temporal_reasoning=any(re.search(p, prompt_lower) for p in temporal_patterns),
        asks_about_patterns=any(re.search(p, prompt_lower) for p in pattern_patterns),
        debugging_task=any(re.search(p, prompt_lower) for p in debug_patterns),
        requires_exhaustive_search=any(re.search(p, prompt_lower) for p in exhaustive_patterns),
        security_review_task=any(re.search(p, prompt_lower) for p in security_review_patterns),
        architecture_analysis=any(re.search(p, prompt_lower) for p in architecture_patterns),
        user_wants_thorough=any(re.search(p, prompt_lower) for p in thorough_patterns),
        user_wants_fast=any(re.search(p, prompt_lower) for p in fast_patterns),
        context_has_multiple_domains=len(context.active_modules) > 2,
        recent_tool_outputs_large=sum(len(o.content) for o in context.tool_outputs[-5:]) > 10000,
        conversation_has_state_changes=_detect_state_changes(context),
        files_span_multiple_modules=len({Path(f).parts[0] for f in context.files if Path(f).parts})
        > 2,
        previous_turn_was_confused=_detect_confusion(context),
        task_is_continuation="continue" in prompt_lower
        or "same" in prompt_lower
        or "also" in prompt_lower[:50],
    )


# Singleton classifier for rlm_core (lazy initialized)
rlm_core_classifier = None


def _get_rlm_core_classifier() -> "rlm_core.PatternClassifier":
    """Get or create the rlm_core PatternClassifier singleton."""
    global rlm_core_classifier
    if rlm_core_classifier is None:
        rlm_core_classifier = rlm_core.PatternClassifier()
    return rlm_core_classifier


def _should_activate_rlm_core(prompt: str, context: SessionContext) -> tuple[bool, str]:
    """
    Delegate activation decision to rlm_core.PatternClassifier.

    This provides consistent behavior with the Go/Rust implementations.
    """
    classifier = _get_rlm_core_classifier()

    # Convert context to rlm_core format
    core_ctx = _convert_to_rlm_core_context(context)

    # Get activation decision from rlm_core
    decision = classifier.should_activate(prompt, core_ctx)

    return decision.should_activate, decision.reason


def should_activate_rlm(
    prompt: str,
    context: SessionContext,
    rlm_mode_forced: bool = False,
    simple_mode_forced: bool = False,
) -> tuple[bool, str]:
    """
    Determine if RLM mode should activate.

    Implements: Spec §6.3 Task Complexity-Based Activation

    Biased toward activation—when in doubt, use RLM.

    Delegates to rlm_core.PatternClassifier for consistent behavior
    across Python and Go implementations.

    Returns:
        (should_activate, reason) tuple
    """
    # Manual overrides (handled before delegation)
    if rlm_mode_forced:
        return True, "manual_override"
    if simple_mode_forced:
        return False, "simple_mode_forced"

    return _should_activate_rlm_core(prompt, context)


def is_definitely_simple(prompt: str, context: SessionContext) -> bool:
    """
    Returns True ONLY for queries that definitely don't need RLM.

    Implements: Spec §6.5 Hybrid Mode

    When in doubt, return False (use RLM).
    """
    prompt_lower = prompt.lower().strip()

    # Explicit simple patterns
    simple_patterns = [
        r"^(show|cat|read|view|open)\s+[\w./]+$",
        r"^(run|execute)\s+(npm|yarn|pnpm|cargo|go|python|pytest)\s+\w+$",
        r"^git\s+(status|log|diff|branch)$",
        r"^what\'?s?\s+in\s+[\w./]+\??$",
        r"^(ok|okay|thanks|got it|understood|yes|no|sure)\.?$",
    ]

    for pattern in simple_patterns:
        if re.match(pattern, prompt_lower) and context.total_tokens < 20000:
            return True

    # Short prompts with no complexity indicators
    if len(prompt) < 50 and context.total_tokens < 10000:
        complexity_words = [
            "why",
            "how",
            "find",
            "fix",
            "bug",
            "error",
            "all",
            "every",
            "change",
            "update",
            "refactor",
            "test",
        ]
        if not any(word in prompt_lower for word in complexity_words):
            return True

    return False


__all__ = [
    "extract_complexity_signals",
    "should_activate_rlm",
    "is_definitely_simple",
]
