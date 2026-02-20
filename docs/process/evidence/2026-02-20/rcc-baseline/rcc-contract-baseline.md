# RCC Contract Baseline

- Generated (UTC): `2026-02-20T17:24:39.047032+00:00`
- Candidate loop SHA: `92046b61f52d78b4f2442ff7822d979569af3ad5`
- Vendor loop SHA: `4bb2f7371e02b5b9c904fb67dcd61eda15725e6a`
- All invariants passing: `True`
- Claim scope: `claim-ready-for-pinned-vendor-sha-only`

| Invariant | Pass | Details |
|---|---|---|
| A1 | PASS | Import path `rlm_core` is usable. |
| A2 | PASS | Trajectory enum variants and constructors used by this repo are compatible. |
| A3 | PASS | Basic memory CRUD/search semantics are compatible. |
| A4 | PASS | Classifier/router delegation paths are callable and structurally compatible. |
| A5 | PASS | Both REPL batched-query helpers are available. |

## Observed Data

```json
{
  "script": "scripts/rcc_contract_probe.py",
  "script_version": 1,
  "generated_utc": "2026-02-20T17:24:39.047032+00:00",
  "repo_root": "/Users/rand/src/rlm-claude-code",
  "python_version": "3.12.11",
  "platform": "macOS-26.3-arm64-arm-64bit",
  "loop_candidate_repo": "/Users/rand/src/loop",
  "loop_candidate_sha": "92046b61f52d78b4f2442ff7822d979569af3ad5",
  "vendor_loop_sha": "4bb2f7371e02b5b9c904fb67dcd61eda15725e6a",
  "all_passed": true,
  "claim_scope": "claim-ready-for-pinned-vendor-sha-only",
  "invariants": [
    {
      "invariant": "A1",
      "title": "Python import path and module usability",
      "passed": true,
      "details": "Import path `rlm_core` is usable.",
      "observed": {
        "missing_exports": [],
        "version": "0.1.0",
        "features": [
          "python"
        ]
      }
    },
    {
      "invariant": "A2",
      "title": "Trajectory enum/constructor compatibility",
      "passed": true,
      "details": "Trajectory enum variants and constructors used by this repo are compatible.",
      "observed": {
        "missing_variants": [],
        "missing_factories": [],
        "factory_errors": {}
      }
    },
    {
      "invariant": "A3",
      "title": "Memory-store behavior compatibility",
      "passed": true,
      "details": "Basic memory CRUD/search semantics are compatible.",
      "observed": {
        "db_path": "/var/folders/vx/_xxfn0z95bv7qmfgb3nx6z700000gn/T/rcc-memory-probe-nteuz7bw/memory.db",
        "node_created": "72e1cb1d-7035-44ef-b655-d1fa7b92f60e",
        "search_hits": 1,
        "deleted": true,
        "stats_total_nodes": 0,
        "stats_total_edges": 0
      }
    },
    {
      "invariant": "A4",
      "title": "Pattern-classifier and smart-router delegation",
      "passed": true,
      "details": "Classifier/router delegation paths are callable and structurally compatible.",
      "observed": {
        "activation_should_activate": true,
        "activation_reason": "Score 13 >= threshold 3 (signals: multi-file reference, cross-context reasoning, debugging, exhaustive search, security review, continuation)",
        "routing_reason": "Query type 'architecture' at depth 0 -> flagship tier (adjusted from flagship)",
        "routing_model": "claude-3-opus-20240229"
      }
    },
    {
      "invariant": "A5",
      "title": "REPL batched-query helper compatibility",
      "passed": true,
      "details": "Both REPL batched-query helpers are available.",
      "observed": {
        "llm_batch": true,
        "llm_query_batched": true
      }
    }
  ]
}
```
