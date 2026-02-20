"""
Integration tests for the complexity-check binary.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _run_complexity_check(prompt: str, env: dict[str, str] | None = None) -> dict[str, str]:
    payload = json.dumps({"user_prompt": prompt})
    result = subprocess.run(
        ["go", "run", "./cmd/complexity-check"],
        input=payload,
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("go") is None, reason="Go toolchain required")
def test_complexity_check_reads_prompt_from_stdin_and_emits_decision() -> None:
    """SPEC-17.09: Binary reads stdin payload and returns activation decision output."""
    output = _run_complexity_check("Debug why auth fails across modules")

    assert output["decision"] == "approve"
    assert "RLM " in output.get("reason", "")


@pytest.mark.skipif(shutil.which("go") is None, reason="Go toolchain required")
def test_complexity_check_honors_rlm_disabled_env() -> None:
    """SPEC-17.09: Binary exits early with neutral approval when RLM_DISABLED is set."""
    env = os.environ.copy()
    env["RLM_DISABLED"] = "1"
    output = _run_complexity_check("Debug why auth fails across modules", env=env)

    assert output == {"decision": "approve"}
