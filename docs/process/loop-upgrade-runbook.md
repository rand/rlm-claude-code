# Loop Pin Upgrade Runbook

Date: 2026-02-20  
Owner epic: `rlm-tfib`  
Task: `rlm-tfib.6`

## Purpose

Define an executable rollout and rollback process for `vendor/loop` upgrades in
`rlm-claude-code`, including required compatibility gates and maintenance
cadence.

## Tuple Terms

- Candidate SHA: `git -C /Users/rand/src/loop rev-parse HEAD`
- Vendored SHA: `git -C /Users/rand/src/rlm-claude-code/vendor/loop rev-parse HEAD`
- Claim scope values:
  - `not-claim-ready`
  - `claim-ready-for-pinned-vendor-sha-only`
  - `claim-ready-for-candidate-tuple`

## Rollout Classes

1. Class A (low risk):
- no pybind API surface changes used by `rlm-claude-code`
- expected to pass A1-A5 unchanged

2. Class B (medium risk):
- pybind/runtime behavior updates in delegated components (memory/router/trajectory/repl)
- requires full A1-A5 + targeted unit slices

3. Class C (high risk):
- contract-affecting changes in REPL protocol, pybind exports, or memory semantics
- requires explicit migration notes + rollback rehearsal before merge

## Required Release Checklist

1. Confirm clean parent and submodule state before pin edits:
```bash
git status --short
git submodule status vendor/loop
git -C vendor/loop status --short
```
2. Update submodule pin intentionally:
```bash
git -C vendor/loop fetch --all --tags
git -C vendor/loop checkout <candidate-sha>
git add vendor/loop
```
3. Run compatibility evidence gates:
```bash
make rcc-contract-test
make rcc-contract-gate
```
4. Run policy gates:
```bash
UV_CACHE_DIR=.uv-cache uv run dp review --json
UV_CACHE_DIR=.uv-cache uv run dp verify --json
UV_CACHE_DIR=.uv-cache uv run dp enforce pre-commit --policy dp-policy.json --json
```
5. Record tuple claim scope from probe output (`all_passed=... claim_scope=...`).

## Rollback Triggers

Rollback immediately when any of the following occur after pin movement:
- `make rcc-contract-gate` fails.
- `import rlm_core` fails or required exports in A1/A2 are missing.
- deterministic unit slices covering REPL/orchestrator contract fail.
- claim scope regresses to `not-claim-ready`.

## Rollback Procedure

1. Move submodule back to last known-good SHA:
```bash
git -C vendor/loop checkout <known-good-sha>
git add vendor/loop
```
2. Re-run gates:
```bash
make rcc-contract-gate
UV_CACHE_DIR=.uv-cache uv run dp review --json
```
3. Commit rollback with tuple metadata in message/body.

## Validated Known-Good Rollback Tuple

- Known-good vendored SHA: `6779cdbc970c70f3ce82a998d6dcda59cd171560`
- Validation command run on 2026-02-20:
```bash
make rcc-contract-gate
```
- Evidence artifact:
`/Users/rand/src/rlm-claude-code/docs/process/evidence/2026-02-20/rcc-baseline/rcc-contract-baseline.json`

## Maintenance Cadence

1. Weekly (owner: `rlm` maintainers):
- check candidate loop SHA drift
- log whether a pin update is needed

2. Monthly (owner: `rlm` maintainers):
- run `make rcc-contract-test` + `make rcc-contract-gate`
- confirm tuple claim scope status in evidence artifacts

3. Quarterly (owner: repo steward):
- execute rollback drill to last known-good tuple on a branch
- verify gate recovery and document results in `docs/process/evidence/<date>/`
