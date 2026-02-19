from __future__ import annotations

class Provider:
    Anthropic: Provider
    OpenAI: Provider
    OpenRouter: Provider
    Google: Provider

class ModelTier:
    Flagship: ModelTier
    Balanced: ModelTier
    Fast: ModelTier

class QueryType:
    Architecture: QueryType
    MultiFile: QueryType
    Debugging: QueryType
    Extraction: QueryType
    Simple: QueryType

    @staticmethod
    def classify(query: str) -> QueryType: ...
    def base_tier(self) -> ModelTier: ...

class ModelSpec:
    id: str
    name: str
    provider: Provider
    tier: ModelTier
    context_window: int
    max_output: int
    input_cost_per_m: float
    output_cost_per_m: float
    supports_caching: bool
    supports_vision: bool
    supports_tools: bool

    def __init__(
        self,
        id: str,
        name: str,
        provider: Provider,
        tier: ModelTier,
        context_window: int,
        max_output: int,
        input_cost: float,
        output_cost: float,
    ) -> None: ...
    @staticmethod
    def claude_opus() -> ModelSpec: ...
    @staticmethod
    def claude_sonnet() -> ModelSpec: ...
    @staticmethod
    def claude_haiku() -> ModelSpec: ...
    @staticmethod
    def gpt4o() -> ModelSpec: ...
    @staticmethod
    def gpt4o_mini() -> ModelSpec: ...
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float: ...

class ChatMessage:
    role: str
    content: str

    @staticmethod
    def system(content: str) -> ChatMessage: ...
    @staticmethod
    def user(content: str) -> ChatMessage: ...
    @staticmethod
    def assistant(content: str) -> ChatMessage: ...

class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None
    cache_creation_tokens: int | None

    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read: int | None = None,
        cache_creation: int | None = None,
    ) -> None: ...
    def total(self) -> int: ...
    def effective_input_tokens(self) -> int: ...

class CompletionRequest:
    model: str | None
    system: str | None
    max_tokens: int | None
    temperature: float | None

    def __init__(self) -> None: ...
    def with_model(self, model: str) -> CompletionRequest: ...
    def with_system(self, system: str) -> CompletionRequest: ...
    def with_message(self, message: ChatMessage) -> CompletionRequest: ...
    def with_max_tokens(self, max_tokens: int) -> CompletionRequest: ...
    def with_temperature(self, temperature: float) -> CompletionRequest: ...
    def with_caching(self, enable: bool) -> CompletionRequest: ...

class CompletionResponse:
    id: str
    model: str
    content: str
    stop_reason: str | None
    usage: TokenUsage
    timestamp: str
    cost: float | None

class RoutingContext:
    depth: int
    max_depth: int
    remaining_budget: float | None

    def __init__(self) -> None: ...
    def with_depth(self, depth: int) -> RoutingContext: ...
    def with_max_depth(self, max_depth: int) -> RoutingContext: ...
    def with_budget(self, budget: float) -> RoutingContext: ...
    def with_provider(self, provider: Provider) -> RoutingContext: ...
    def requiring_caching(self) -> RoutingContext: ...
    def requiring_vision(self) -> RoutingContext: ...
    def requiring_tools(self) -> RoutingContext: ...

class SmartRouter:
    def __init__(self) -> None: ...
    def route(self, query: str, context: RoutingContext) -> RoutingDecision: ...
    def models(self) -> list[ModelSpec]: ...

class RoutingDecision:
    model: ModelSpec
    query_type: QueryType
    tier: ModelTier
    reason: str
    estimated_cost: float | None

class CostTracker:
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cost: float
    request_count: int
    by_model: dict[str, dict[str, float]]

    def __init__(self) -> None: ...
    def record(self, model: str, usage: TokenUsage, cost: float | None = None) -> None: ...
    def merge(self, other: CostTracker) -> None: ...
