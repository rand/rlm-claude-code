#!/usr/bin/env python3
"""Probe A1-A5 consumer-contract compatibility for rlm-claude-code.

This script produces reproducible compatibility artifacts that include:
- loop candidate SHA (from standalone loop repo, if available)
- vendor/loop SHA (the effective pinned runtime in this repo)
- invariant-level outcomes for A1..A5

Use `--strict` to return non-zero on any invariant failure.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class InvariantResult:
    invariant: str
    title: str
    passed: bool
    details: str
    observed: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_sha(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _run_a1() -> InvariantResult:
    try:
        import rlm_core

        required = [
            "version",
            "available_features",
            "MemoryStore",
            "PatternClassifier",
            "SmartRouter",
            "TrajectoryEvent",
        ]
        missing = [name for name in required if not hasattr(rlm_core, name)]
        version = rlm_core.version() if hasattr(rlm_core, "version") else None
        features = (
            rlm_core.available_features() if hasattr(rlm_core, "available_features") else None
        )
        passed = not missing and version is not None
        details = "Import path `rlm_core` is usable." if passed else "Missing required exports."
        return InvariantResult(
            invariant="A1",
            title="Python import path and module usability",
            passed=passed,
            details=details,
            observed={"missing_exports": missing, "version": version, "features": features},
        )
    except Exception as exc:  # pragma: no cover - defensive probe path
        return InvariantResult(
            invariant="A1",
            title="Python import path and module usability",
            passed=False,
            details=f"Import failed: {exc}",
            observed={"error": repr(exc)},
        )


def _run_a2() -> InvariantResult:
    try:
        import rlm_core

        required_enum_variants = [
            "RlmStart",
            "Analyze",
            "ReplExec",
            "ReplResult",
            "Reason",
            "RecurseStart",
            "RecurseEnd",
            "Final",
            "Error",
            "ToolUse",
            "CostReport",
            "BudgetComputed",
            "VerifyComplete",
        ]
        missing_variants = [
            name
            for name in required_enum_variants
            if not hasattr(rlm_core.TrajectoryEventType, name)
        ]

        factory_calls: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {
            "rlm_start": (("query",), {}),
            "analyze": (("analysis",), {"depth": 0}),
            "repl_exec": ((0, "x = 1"), {}),
            "repl_result": ((0, "ok"), {}),
            "reason": ((0, "reasoning"), {}),
            "recurse_start": ((1, "child query"), {}),
            "recurse_end": ((1, "child result"), {}),
            "final_answer": (("final answer",), {"depth": 0}),
            "error": ((0, "error"), {}),
        }

        missing_factories = [
            name for name in factory_calls if not hasattr(rlm_core.TrajectoryEvent, name)
        ]
        factory_errors: dict[str, str] = {}
        for name, (args, kwargs) in factory_calls.items():
            if name in missing_factories:
                continue
            try:
                event = getattr(rlm_core.TrajectoryEvent, name)(*args, **kwargs)
                _ = event.to_json()
            except Exception as exc:  # pragma: no cover - defensive probe path
                factory_errors[name] = repr(exc)

        passed = not missing_variants and not missing_factories and not factory_errors
        if passed:
            details = "Trajectory enum variants and constructors used by this repo are compatible."
        else:
            details = "Trajectory compatibility mismatch detected."
        return InvariantResult(
            invariant="A2",
            title="Trajectory enum/constructor compatibility",
            passed=passed,
            details=details,
            observed={
                "missing_variants": missing_variants,
                "missing_factories": missing_factories,
                "factory_errors": factory_errors,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive probe path
        return InvariantResult(
            invariant="A2",
            title="Trajectory enum/constructor compatibility",
            passed=False,
            details=f"Probe failed: {exc}",
            observed={"error": repr(exc)},
        )


def _run_a3() -> InvariantResult:
    try:
        import rlm_core

        tmpdir = Path(tempfile.mkdtemp(prefix="rcc-memory-probe-"))
        db_path = tmpdir / "memory.db"
        try:
            store = rlm_core.MemoryStore.open(str(db_path))
            node = rlm_core.Node(
                rlm_core.NodeType.Fact,
                "rcc probe fact",
                tier=rlm_core.Tier.Task,
                confidence=0.9,
                metadata={"probe": "a3"},
            )
            node_id = store.add_node(node)
            fetched = store.get_node(node_id)
            hits = store.search_content("rcc probe fact", 10)
            deleted = store.delete_node(node_id)
            stats = store.stats()
            passed = fetched is not None and bool(hits) and bool(deleted)
            details = (
                "Basic memory CRUD/search semantics are compatible."
                if passed
                else "Memory semantics probe failed."
            )
            return InvariantResult(
                invariant="A3",
                title="Memory-store behavior compatibility",
                passed=passed,
                details=details,
                observed={
                    "db_path": str(db_path),
                    "node_created": node_id,
                    "search_hits": len(hits),
                    "deleted": bool(deleted),
                    "stats_total_nodes": stats.total_nodes,
                    "stats_total_edges": stats.total_edges,
                },
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as exc:  # pragma: no cover - defensive probe path
        return InvariantResult(
            invariant="A3",
            title="Memory-store behavior compatibility",
            passed=False,
            details=f"Probe failed: {exc}",
            observed={"error": repr(exc)},
        )


def _run_a4() -> InvariantResult:
    try:
        import rlm_core

        ctx = rlm_core.SessionContext()
        ctx.add_user_message("review auth and db interactions")
        classifier = rlm_core.PatternClassifier()
        decision = classifier.should_activate("find all auth issues across modules", ctx)

        router = rlm_core.SmartRouter()
        routing_ctx = (
            rlm_core.RoutingContext().with_depth(0).with_max_depth(3).with_budget(1.0)
        )
        routing = router.route("analyze architecture and boundaries", routing_ctx)

        passed = all(
            [
                hasattr(decision, "should_activate"),
                hasattr(decision, "reason"),
                hasattr(routing, "model"),
                hasattr(routing, "reason"),
            ]
        )
        details = (
            "Classifier/router delegation paths are callable and structurally compatible."
            if passed
            else "Classifier/router delegation probe failed."
        )
        return InvariantResult(
            invariant="A4",
            title="Pattern-classifier and smart-router delegation",
            passed=passed,
            details=details,
            observed={
                "activation_should_activate": bool(getattr(decision, "should_activate", False)),
                "activation_reason": str(getattr(decision, "reason", "")),
                "routing_reason": str(getattr(routing, "reason", "")),
                "routing_model": str(getattr(getattr(routing, "model", None), "id", "")),
            },
        )
    except Exception as exc:  # pragma: no cover - defensive probe path
        return InvariantResult(
            invariant="A4",
            title="Pattern-classifier and smart-router delegation",
            passed=False,
            details=f"Probe failed: {exc}",
            observed={"error": repr(exc)},
        )


def _run_a5(repo_root: Path) -> InvariantResult:
    try:
        import sys

        sys.path.insert(0, str(repo_root))
        import src.repl_environment as repl_environment
        from src.types import SessionContext

        # Avoid loading optional heavy plotting/data tooling during a contract probe.
        original_add_extended = repl_environment.RLMEnvironment._add_extended_tooling
        repl_environment.RLMEnvironment._add_extended_tooling = lambda self: None
        try:
            env = repl_environment.RLMEnvironment(SessionContext(), use_restricted=False)
        finally:
            repl_environment.RLMEnvironment._add_extended_tooling = original_add_extended
        batch = callable(env.globals.get("llm_batch"))
        alias = callable(env.globals.get("llm_query_batched"))
        passed = batch and alias
        details = (
            "Both REPL batched-query helpers are available."
            if passed
            else "Missing required helper(s) for migration-window compatibility."
        )
        return InvariantResult(
            invariant="A5",
            title="REPL batched-query helper compatibility",
            passed=passed,
            details=details,
            observed={
                "llm_batch": batch,
                "llm_query_batched": alias,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive probe path
        return InvariantResult(
            invariant="A5",
            title="REPL batched-query helper compatibility",
            passed=False,
            details=f"Probe failed: {exc}",
            observed={"error": repr(exc)},
        )


def _to_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append("# RCC Contract Baseline")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_utc']}`")
    lines.append(f"- Candidate loop SHA: `{report['loop_candidate_sha']}`")
    lines.append(f"- Vendor loop SHA: `{report['vendor_loop_sha']}`")
    lines.append(f"- All invariants passing: `{report['all_passed']}`")
    lines.append(f"- Claim scope: `{report['claim_scope']}`")
    lines.append("")
    lines.append("| Invariant | Pass | Details |")
    lines.append("|---|---|---|")
    for item in report["invariants"]:
        pass_cell = "PASS" if item["passed"] else "FAIL"
        details = item["details"].replace("\n", " ").strip()
        lines.append(f"| {item['invariant']} | {pass_cell} | {details} |")
    lines.append("")
    lines.append("## Observed Data")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _compute_claim_scope(all_passed: bool, candidate_sha: str | None, vendor_sha: str | None) -> str:
    if not all_passed:
        return "not-claim-ready"
    if candidate_sha and vendor_sha and candidate_sha == vendor_sha:
        return "claim-ready-for-candidate-tuple"
    return "claim-ready-for-pinned-vendor-sha-only"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact output directory (default: docs/process/evidence/<date>/rcc-baseline).",
    )
    parser.add_argument(
        "--candidate-loop-repo",
        type=Path,
        default=Path("/Users/rand/src/loop"),
        help="Path to standalone loop repo for candidate SHA capture.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any invariant fails.",
    )
    args = parser.parse_args()

    root = _repo_root()
    today = datetime.now(UTC).date().isoformat()
    out_dir = (
        args.output_dir
        if args.output_dir is not None
        else root / "docs" / "process" / "evidence" / today / "rcc-baseline"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    vendor_sha = _git_sha(root / "vendor" / "loop")
    candidate_sha = _git_sha(args.candidate_loop_repo)

    invariants = [
        _run_a1(),
        _run_a2(),
        _run_a3(),
        _run_a4(),
        _run_a5(root),
    ]
    all_passed = all(item.passed for item in invariants)
    claim_scope = _compute_claim_scope(all_passed, candidate_sha, vendor_sha)

    report = {
        "script": "scripts/rcc_contract_probe.py",
        "script_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "repo_root": str(root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "loop_candidate_repo": str(args.candidate_loop_repo),
        "loop_candidate_sha": candidate_sha,
        "vendor_loop_sha": vendor_sha,
        "all_passed": all_passed,
        "claim_scope": claim_scope,
        "invariants": [asdict(item) for item in invariants],
    }

    json_path = out_dir / "rcc-contract-baseline.json"
    md_path = out_dir / "rcc-contract-baseline.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"all_passed={all_passed} claim_scope={claim_scope}")

    if args.strict and not all_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
