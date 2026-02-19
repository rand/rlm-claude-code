from __future__ import annotations

from rlm_core._context import SessionContext

class ActivationDecision:
    should_activate: bool
    reason: str
    score: int

class PatternClassifier:
    def __init__(self) -> None: ...
    def should_activate(self, query: str, context: SessionContext) -> ActivationDecision: ...
