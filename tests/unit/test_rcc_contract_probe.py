"""
Contract tests for A1-A5 compatibility probe and tuple evidence metadata.
"""

import json
import subprocess
import sys
from pathlib import Path


def _run_probe(tmp_path: Path, strict: bool = False) -> tuple[int, dict]:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rcc_contract_probe.py"
    out_dir = tmp_path / "rcc-probe"
    cmd = [sys.executable, str(script), "--output-dir", str(out_dir)]
    if strict:
        cmd.append("--strict")

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    report_path = out_dir / "rcc-contract-baseline.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = {}

    return proc.returncode, report


def test_probe_emits_a1_a5_results_and_tuple_metadata(tmp_path):
    exit_code, report = _run_probe(tmp_path, strict=False)

    assert exit_code == 0
    assert "loop_candidate_sha" in report
    assert "vendor_loop_sha" in report
    assert "claim_scope" in report
    assert "invariants" in report

    by_id = {item["invariant"]: item for item in report["invariants"]}
    assert set(by_id.keys()) == {"A1", "A2", "A3", "A4", "A5"}


def test_probe_strict_mode_requires_all_invariants_to_pass(tmp_path):
    exit_code, report = _run_probe(tmp_path, strict=True)

    assert exit_code == 0
    assert report["all_passed"] is True
    for invariant in report["invariants"]:
        assert invariant["passed"] is True, invariant["details"]
