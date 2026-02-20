# RCC Consumer Contract Baseline (A1-A5)

Date: 2026-02-20  
Owner epic: `rlm-tfib`  
Tasks: `rlm-tfib.1`, `rlm-tfib.5`

## Purpose

Define the repository-local interpretation of consumer invariants A1-A5 for
`rlm-claude-code` and make compatibility claims reproducible, pin-aware, and
auditable.

Primary upstream contract reference:
- `loop/docs/execution-plan/contracts/CONSUMER-INTEGRATION.md`

## Invariants

## A1: Import Path and Module Name

`rlm_core` must remain importable and usable from this repository as the
runtime module name.

Minimum required surface:
- `version()`, `available_features()`
- `MemoryStore`
- `PatternClassifier`
- `SmartRouter`
- `TrajectoryEvent`

## A2: Trajectory Enum and Constructor Compatibility

Surfaces consumed by `/Users/rand/src/rlm-claude-code/src/trajectory.py` must
remain structurally compatible:
- event enum variants used in mappings,
- factory methods used for event creation.

## A3: Memory-Store Behavior Compatibility

Behavior relied on by `/Users/rand/src/rlm-claude-code/src/memory_store.py`
must remain compatible for baseline node CRUD and search semantics.

This baseline probe is not a full concurrency/WAL proof; it is a deterministic
contract smoke check used to detect immediate breakage before deeper gates.

## A4: Classifier and Router Delegation

Delegation paths used by:
- `/Users/rand/src/rlm-claude-code/src/complexity_classifier.py`
- `/Users/rand/src/rlm-claude-code/src/smart_router.py`

must remain callable and return structurally compatible decisions.

## A5: REPL Batched-Query Helper Compatibility

During migration windows, both helper names must resolve:
- canonical: `llm_batch`
- compatibility alias: `llm_query_batched`

This prevents orchestration/repl behavior drift while runtime contracts are
being reconciled.

## Claim Policy (Pin-Aware)

Compatibility claims are accepted only when produced by the local harness:
- `/Users/rand/src/rlm-claude-code/scripts/rcc_contract_probe.py`

Every claim artifact MUST include:
- standalone loop candidate SHA,
- `vendor/loop` SHA used by this repo,
- A1-A5 invariant outcomes.

Claim-scope rules:
1. If any invariant fails: scope is `not-claim-ready`.
2. If all invariants pass and SHAs differ: scope is
   `claim-ready-for-pinned-vendor-sha-only`.
3. If all invariants pass and SHAs match: scope is
   `claim-ready-for-candidate-tuple`.

## Commands

Generate baseline artifacts:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/rcc_contract_probe.py
```

Generate artifacts at an explicit location:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/rcc_contract_probe.py \
  --output-dir docs/process/evidence/2026-02-20/rcc-baseline
```

Strict gate (non-zero on any failure):

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/rcc_contract_probe.py --strict
```

Run automated A1-A5 probe tests:

```bash
make rcc-contract-test
```

## Artifact Locations

Default output directory:
- `/Users/rand/src/rlm-claude-code/docs/process/evidence/<YYYY-MM-DD>/rcc-baseline/`

Produced files:
- `rcc-contract-baseline.json`
- `rcc-contract-baseline.md`
