# RCC Contract Baseline

- Generated (UTC): `2026-02-20T16:56:16.752398+00:00`
- Candidate loop SHA: `4bb2f7371e02b5b9c904fb67dcd61eda15725e6a`
- Vendor loop SHA: `4bb2f7371e02b5b9c904fb67dcd61eda15725e6a`
- All invariants passing: `True`
- Claim scope: `claim-ready-for-candidate-tuple`

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
  "generated_utc": "2026-02-20T16:56:16.752398+00:00",
  "repo_root": "/Users/rand/src/rlm-claude-code",
  "python_version": "3.12.11",
  "platform": "macOS-26.3-arm64-arm-64bit",
  "loop_candidate_repo": "/Users/rand/src/loop",
  "loop_candidate_sha": "4bb2f7371e02b5b9c904fb67dcd61eda15725e6a",
  "vendor_loop_sha": "4bb2f7371e02b5b9c904fb67dcd61eda15725e6a",
  "all_passed": true,
  "claim_scope": "claim-ready-for-candidate-tuple",
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
        "db_path": "/var/folders/vx/_xxfn0z95bv7qmfgb3nx6z700000gn/T/rcc-memory-probe-5ptcu3kh/memory.db",
        "node_created": "5e853b5f-18bb-46c7-bf8f-3a13558d2ea7",
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
