# Loop Post-M7 Adoption Plan (dp-codex)

Date: 2026-02-20  
Planning task: `rlm-claude-code-8it.1`  
Execution epic: `rlm-tfib`

## 1. Objective

Update `rlm-claude-code` to fully leverage the current `loop` project **within current Python binding limits**.

"Fully leverage" in this plan means:
- adopt all relevant post-M7 runtime/protocol fixes that are already available to this consumer,
- remove known contract drifts (A1-A5) between `rlm-claude-code` and `loop`,
- keep behavior pin-aware and evidence-driven,
- preserve compatibility and rollback safety.

This does **not** mean a full Rust-orchestrator swap, because those surfaces are still not exposed in Python bindings.

## 2. Deep-Dive Baseline (Research Summary)

### 2.1 Consumer and upstream snapshots

- `rlm-claude-code` vendored `loop` SHA: `6779cdbc970c70f3ce82a998d6dcda59cd171560`
- standalone `loop` working SHA (research baseline): `4bb2f7371e02b5b9c904fb67dcd61eda15725e6a`
- `loop` execution tracker marks M0..M7 complete as of 2026-02-20.

### 2.2 Upstream policy constraints that govern this migration

From loop execution/contract docs:
- compatibility for `rlm-claude-code` is release-blocking in affected windows,
- compatibility claims are **pin-aware** (candidate SHA vs vendored SHA scope),
- current migration scope is component delegation; full swap is explicitly deferred,
- consumer invariants A1-A5 remain mandatory.

### 2.3 Observed integration state in rlm-claude-code

Current delegation is real and active in:
- complexity classification (`PatternClassifier`),
- memory (`MemoryStore`),
- trajectory (`TrajectoryEvent`),
- epistemic claim extraction,
- optional smart-router delegation.

Core orchestration and REPL control remain Python-owned.

### 2.4 Concrete drift/gap inventory

1. REPL helper compatibility drift:
- local REPL environment exposes `llm_batch` but not `llm_query_batched` alias.

2. Typed-submit protocol drift:
- local flow is centered on `FINAL:` / `FINAL_VAR:` parsing;
- no equivalent for upstream signature registration + `SUBMIT(...)` result path.

3. Binding-surface packaging/stub drift risk:
- local `rlm_core` package layout/stubs differ from upstream Python package structure.

4. Validation drift risk:
- no explicit A1-A5 compatibility gate suite in this repo today;
- no local tuple evidence workflow that emits both candidate/pin scope metadata.

5. Vendored gap size is non-trivial:
- `rlm-core/src`: `97 files changed, 8401 insertions, 1843 deletions` (no-index diff),
- `rlm-core/python/rlm_repl`: `17 files changed, 422 insertions, 2 deletions`,
- `rlm-core/python/rlm_core`: `6 files changed, 49 insertions`.

## 3. Migration Principles

1. Contract-first over code-first:
- every change is tied to A1-A5 and consumer integration contract semantics.

2. Pin-aware truthfulness:
- compatibility claims must always include both tuple SHAs and claim scope.

3. Delegation-first architecture:
- harvest all available upstream value without pretending unavailable Python surfaces exist.

4. Backward-safe rollout:
- preserve legacy execution path while introducing typed-submit path; remove only after evidence.

5. Measurable gates before closure:
- no phase is complete without deterministic tests and reproducible artifacts.

## 4. Target End State

At completion of this epic:
- vendored loop pin is intentionally upgraded to approved post-M7 tuple,
- REPL/orchestrator compatibility with upstream helper/protocol behavior is restored,
- local `rlm_core` wrappers/stubs are aligned with vendored runtime surface,
- A1-A5 gates run in CI/local with tuple evidence output,
- release/rollback runbook and cadence are documented and operational.

## 5. Phased Execution Plan

## Phase 0: Baseline and Gate Harness (`rlm-tfib.1`)

Deliverables:
- Add local consumer contract baseline doc for A1-A5 interpretation in this repo.
- Add compatibility harness command(s) that always emit:
  - loop candidate SHA,
  - `vendor/loop` SHA,
  - gate outcomes.
- Define claim scope language: "verified for pinned vendor SHA" vs "verified for candidate tuple".

Exit criteria:
- A1-A5 baseline is documented.
- Harness run output is reproducible locally.

## Phase 1: Pin + Binding Surface Sync (`rlm-tfib.4`)

Deliverables:
- Update `vendor/loop` submodule to approved post-M7 candidate.
- Reconcile local `rlm_core` Python package wrapper/stubs to eliminate avoidable drift from vendored package shape.
- Document any intentionally retained local deviations.

Exit criteria:
- `import rlm_core` works with upgraded pin.
- Local wrapper/stub surface is traceably aligned to vendored runtime package.

## Phase 2: REPL Protocol Parity (`rlm-tfib.3`)

Deliverables:
- Add migration-window compatibility behavior for batched helper naming (`llm_batch` canonical + compatibility alias).
- Introduce signature registration/clear flow and `SUBMIT(...)` handling in local orchestration boundary.
- Extend parser behavior so typed-submit flow is first-class while keeping legacy `FINAL` fallback.

Exit criteria:
- Typed-submit scenarios pass deterministic tests.
- Legacy `FINAL` path still works.

## Phase 3: Deferred Bridge Alignment (`rlm-tfib.2`)

Deliverables:
- Align pending-operation resolution semantics with upstream runtime behavior.
- Handle `submit_result` validation outcomes with actionable user-visible errors.
- Add structured telemetry/events for deferred and batch resolution lifecycle.

Exit criteria:
- deferred and batch flows are deterministic in tests,
- submit validation failures are surfaced clearly,
- no regression in existing orchestration flows.

## Phase 4: A1-A5 Contract Suite + Tuple Evidence (`rlm-tfib.5`)

Deliverables:
- Implement explicit A1-A5 tests:
  - A1 import/module usability,
  - A2 enum/constructor compatibility,
  - A3 memory behavior/locking assumptions,
  - A4 classifier/router delegation behavior,
  - A5 batched helper compatibility.
- Add evidence pipeline commands that capture tuple metadata + gate result summary.

Exit criteria:
- suite fails on contract drift,
- artifacts are adequate for compatibility claim review.

## Phase 5: Rollout and Governance (`rlm-tfib.6`)

Deliverables:
- Repo-specific rollout + rollback playbook for loop pin upgrades.
- Required gate checklist for release.
- Ongoing maintenance cadence ownership and schedule.
- Runbook location: `/Users/rand/src/rlm-claude-code/docs/process/loop-upgrade-runbook.md`.

Exit criteria:
- runbook is complete and executable,
- rollback to prior tuple is documented and validated.

## 6. Beads Execution Backlog (Created)

- `rlm-tfib.1` Establish loop compatibility baseline and pin-aware gate harness
- `rlm-tfib.4` Sync vendor/loop pin and refresh Python binding surface (depends on `.1`)
- `rlm-tfib.3` Implement REPL protocol parity with loop post-M7
- `rlm-tfib.2` Align orchestrator deferred execution bridge (depends on `.3`)
- `rlm-tfib.5` Build A1-A5 contract suite and tuple evidence pipeline (depends on `.2`)
- `rlm-tfib.6` Publish rollout/rollback/maintenance cadence (depends on `.5`)

Planning task:
- `rlm-claude-code-8it.1` (this document + backlog creation)

## 7. Risks and Controls

1. Runtime breakage from pin jump
- Control: staged rollout + A1-A5 gates + tuple-scoped claims.

2. Hidden protocol regressions (typed submit vs FINAL)
- Control: dual-path tests and explicit fallback retention until evidence is stable.

3. SQLite behavior changes in memory layer
- Control: dedicated A3 compatibility tests and WAL/locking focused regression checks.

4. Drift reappearing after initial migration
- Control: recurring compatibility cadence and release checklist hard-gates.

## 8. dp-codex Execution Method

Plan preparation followed dp-codex conventions by:
- claiming tracked work in beads,
- creating a dedicated epic and phased child tasks with dependencies,
- tying each phase to explicit acceptance criteria and gate expectations,
- grounding scope in consumer contracts and tuple-based compatibility policy.

Note: `uv run dp ...` CLI was not available in this repository environment during planning, so beads-native decomposition and contract-based planning were used as the authoritative dp-codex workflow substrate.

## 9. Source Artifacts Used in This Deep Dive

`rlm-claude-code`:
- `pyproject.toml`
- `src/repl_environment.py`
- `src/response_parser.py`
- `src/orchestrator/core.py`
- `src/memory_store.py`
- `src/complexity_classifier.py`
- `src/trajectory.py`
- `src/smart_router.py`
- `src/epistemic/claim_extractor.py`
- `rlm_core/__init__.py`
- `rlm_core/__init__.pyi`

`loop` / `rlm-core`:
- `docs/execution-plan/STATUS.md`
- `docs/execution-plan/DECISIONS.md`
- `docs/execution-plan/contracts/CONSUMER-INTEGRATION.md`
- `docs/migration-spec-rlm-claude-code.md`
- `docs/internals/python-orchestrator-swap-analysis.md`
- `rlm-core/src/pybind/mod.rs`
- `rlm-core/src/repl.rs`
- `rlm-core/src/adapters/claude_code/adapter.rs`
- `rlm-core/src/orchestrator.rs`
- `rlm-core/python/rlm_repl/main.py`
- `rlm-core/python/rlm_repl/sandbox.py`
- `rlm-core/python/rlm_repl/helpers.py`
- `rlm-core/python/rlm_core/__init__.py`
- `rlm-core/python/rlm_core/__init__.pyi`
