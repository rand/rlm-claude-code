"""Cross-plugin event emission and consumption for RLM."""

from .consume import get_dp_phase, get_rlm_mode, read_latest_event
from .emit import emit_event

__all__ = ["emit_event", "read_latest_event", "get_dp_phase", "get_rlm_mode"]
