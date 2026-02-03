"""Tests for Python event helpers."""

import json
import os
from unittest import mock

import pytest


@pytest.fixture
def events_dir(tmp_path):
    ev_dir = tmp_path / ".claude" / "events"
    ev_dir.mkdir(parents=True)
    with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
        # Patch EVENTS_DIR in both modules
        import src.events.consume as consume_mod
        import src.events.emit as emit_mod

        old_emit = emit_mod.EVENTS_DIR
        old_consume = consume_mod.EVENTS_DIR
        emit_mod.EVENTS_DIR = ev_dir
        consume_mod.EVENTS_DIR = ev_dir
        yield ev_dir
        emit_mod.EVENTS_DIR = old_emit
        consume_mod.EVENTS_DIR = old_consume


def test_emit_creates_files(events_dir):
    from src.events.emit import emit_event

    emit_event({"type": "test", "value": 42}, "test-source")

    log = events_dir / "test-source-events.jsonl"
    assert log.exists()
    data = json.loads(log.read_text().strip())
    assert data["type"] == "test"
    assert "timestamp" in data

    latest = events_dir / "test-source-latest.json"
    assert latest.exists()


def test_read_latest(events_dir):
    from src.events.consume import read_latest_event
    from src.events.emit import emit_event

    emit_event({"type": "first"}, "test-src")
    emit_event({"type": "second"}, "test-src")

    latest = read_latest_event("test-src")
    assert latest["type"] == "second"


def test_read_latest_missing(events_dir):
    from src.events.consume import read_latest_event

    assert read_latest_event("nonexistent") is None


def test_get_dp_phase(events_dir):
    from src.events.consume import get_dp_phase
    from src.events.emit import emit_event

    assert get_dp_phase() == "unknown"
    emit_event({"type": "phase_transition", "to_phase": "spec"}, "disciplined-process")
    assert get_dp_phase() == "spec"


def test_get_rlm_mode(events_dir):
    from src.events.consume import get_rlm_mode
    from src.events.emit import emit_event

    assert get_rlm_mode() == "unknown"
    emit_event({"type": "mode_change", "to_mode": "thorough"}, "rlm-claude-code")
    assert get_rlm_mode() == "thorough"
