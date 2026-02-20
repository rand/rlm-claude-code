# Full-System Empirical Validation Report

Date: 2026-02-20  
Epic: `rlm-bl1c`  
Scope: entire `rlm-claude-code` system surface, including loop (`rlm-core`) consumer contract compatibility.

## Validation Commands and Outcomes

| Command | Purpose | Outcome |
|---|---|---|
| `UV_CACHE_DIR=.uv-cache uv run --extra dev pytest -q` | Full Python correctness and integration suite | `3268 passed, 3 deselected, 8 warnings` |
| `go test ./...` | Go-side classifier/hook correctness | all packages passing |
| `make rcc-contract-gate` | Loop consumer contract gate (A1-A5) | pass, `all_passed=True`, `claim_scope=claim-ready-for-pinned-vendor-sha-only` |
| `UV_CACHE_DIR=.uv-cache uv run dp enforce pre-commit --policy dp-policy.json --json` | Policy pre-commit gate | `ok=true` |
| `UV_CACHE_DIR=.uv-cache uv run dp review --json` | Review gate | `ok=true` |
| `UV_CACHE_DIR=.uv-cache uv run dp verify --json` | Verification gate | `ok=true` |
| `UV_CACHE_DIR=.uv-cache uv run dp enforce pre-push --policy dp-policy.json --json` | Policy pre-push gate | `ok=true` |

Contract artifacts were refreshed at:
- `docs/process/evidence/2026-02-20/rcc-baseline/rcc-contract-baseline.json`
- `docs/process/evidence/2026-02-20/rcc-baseline/rcc-contract-baseline.md`

## JTBD/OODA Empirical Matrix

Empirical OODA coverage is exercised by:
- `tests/integration/test_jtbd_ooda_flows.py`

| JTBD | Observe (signals) | Orient/Decide (strategy) | Act (strategy hints exercised) |
|---|---|---|---|
| JTBD-1 Debug complex multi-layer issues | `debugging_task`, `requires_cross_context_reasoning` | `RECURSIVE_DEBUG` | `map_reduce()`-based synthesis hints |
| JTBD-2 Understand unfamiliar architecture | `architecture_analysis`, `references_multiple_files` | `DISCOVERY` | `find_relevant()` discovery hints |
| JTBD-3 Comprehensive cross-codebase change | `requires_exhaustive_search`, `references_multiple_files` | `EXHAUSTIVE_SEARCH` | `llm_batch()` exhaustive hints |
| JTBD-4 Architectural decision/tradeoffs | `architecture_analysis` | `ARCHITECTURE` | tradeoff-oriented architecture hints |
| JTBD-5 Security/quality completeness | `requires_exhaustive_search`, `security_review_task` | `MAP_REDUCE` | `map_reduce()` completeness hints |
| JTBD-6 Resume/continue prior work | `task_is_continuation`, `previous_turn_was_confused` | `CONTINUATION` | `memory_query()` continuation hints |
| JTBD-7 Fast simple request | simple/conversational | bypass (`simple_task`/`conversational`) | direct response path |

## Unfinished-Implementation Closure in This Execution

1. `find_relevant(..., use_llm_scoring=True)` no-op branch was replaced with real deferred scoring batch creation.
- File: `src/repl_environment.py`

2. Deferred bridge now performs actual map-reduce reduction and relevance reranking post-resolution.
- Files: `src/orchestrator/core.py`, `src/orchestrator.py`

3. Regression coverage added for these behaviors.
- Files: `tests/unit/test_orchestrator_deferred_bridge.py`, `tests/unit/test_repl_advanced_functions.py`

4. Deterministic fallback test was fixed to avoid cascade-path nondeterminism.
- File: `tests/unit/test_local_orchestrator.py`

## Spec Traceability Snapshot

Snapshot script results:
- Total spec IDs in `docs/spec`: `422`
- IDs with direct test trace references: `320`
- IDs without direct test trace references: `102`

Largest uncovered buckets:
- `SPEC-15`: 38 IDs (documented deferred lean-REPL scope)
- `SPEC-14`: 17 IDs
- `SPEC-17`: 12 IDs (mostly covered by Go tests rather than Python trace tags)
- `SPEC-13`: 10 IDs

Interpretation:
- Core system behavior is empirically green end-to-end.
- Remaining traceability work is primarily explicit trace-tag coverage/documentation alignment rather than failing behavior.

## Remaining Follow-Up Work (Tracked Separately)

1. Add a repo-level `make check` target or align AGENTS defaults to existing gate commands.
2. Expand explicit test trace tags for non-deferred uncovered IDs in SPEC-13/14/16/17.
3. Add a time-bounded benchmark execution mode for CI/sandbox reliability (`tests/benchmarks` can be long-running in restricted environments).
