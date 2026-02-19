from __future__ import annotations

from typing import Any

class TrajectoryEventType:
    RlmStart: TrajectoryEventType
    Analyze: TrajectoryEventType
    ReplExec: TrajectoryEventType
    ReplResult: TrajectoryEventType
    Reason: TrajectoryEventType
    RecurseStart: TrajectoryEventType
    RecurseEnd: TrajectoryEventType
    Final: TrajectoryEventType
    Error: TrajectoryEventType
    ToolUse: TrajectoryEventType
    CostReport: TrajectoryEventType
    VerifyStart: TrajectoryEventType
    ClaimExtracted: TrajectoryEventType
    EvidenceChecked: TrajectoryEventType
    BudgetComputed: TrajectoryEventType
    HallucinationFlag: TrajectoryEventType
    VerifyComplete: TrajectoryEventType
    Memory: TrajectoryEventType
    Externalize: TrajectoryEventType
    Decompose: TrajectoryEventType
    Synthesize: TrajectoryEventType
    AdversarialStart: TrajectoryEventType
    CriticInvoked: TrajectoryEventType
    IssueFound: TrajectoryEventType
    AdversarialComplete: TrajectoryEventType

class TrajectoryEvent:
    event_type: TrajectoryEventType
    depth: int
    content: str
    timestamp: str

    def __init__(
        self,
        event_type: TrajectoryEventType,
        content: str,
        depth: int = 0,
    ) -> None: ...
    @staticmethod
    def rlm_start(query: str) -> TrajectoryEvent: ...
    @staticmethod
    def analyze(analysis: str, depth: int = 0) -> TrajectoryEvent: ...
    @staticmethod
    def repl_exec(depth: int, code: str) -> TrajectoryEvent: ...
    @staticmethod
    def repl_result(depth: int, result: str, success: bool = True) -> TrajectoryEvent: ...
    @staticmethod
    def reason(depth: int, reasoning: str) -> TrajectoryEvent: ...
    @staticmethod
    def recurse_start(depth: int, query: str) -> TrajectoryEvent: ...
    @staticmethod
    def recurse_end(depth: int, result: str) -> TrajectoryEvent: ...
    @staticmethod
    def final_answer(answer: str, depth: int = 0) -> TrajectoryEvent: ...
    @staticmethod
    def error(depth: int, error: str) -> TrajectoryEvent: ...
    @staticmethod
    def from_json(json: str) -> TrajectoryEvent: ...
    def with_metadata(self, key: str, value: Any) -> TrajectoryEvent: ...
    def get_metadata(self, key: str) -> Any: ...
    def log_line(self) -> str: ...
    def to_json(self) -> str: ...
