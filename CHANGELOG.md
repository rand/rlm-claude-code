# Changelog

## [0.7.0] - 2026-02-02

### Changed
- **rlm-core is now a required bundled dependency** — no more optional/fallback mode
- `USE_RLM_CORE` / `RLM_USE_CORE` environment variable removed (always enabled)
- `use_rlm_core` config option removed from `rlm-config.json`
- Memory store delegates node CRUD to rlm_core's Rust SQLite backend
- Build system uses maturin for rlm-core PyO3 bindings
- rlm-core vendored as git submodule at `vendor/loop`

### Added
- `id` parameter on `rlm_core.Node()` constructor for creating nodes with pre-existing UUIDs
- `update_fields()` method on `rlm_core.MemoryStore` for field-level updates without Node immutability issues

### Fixed
- Node immutability crash (`updated.id = node_id`) by using `update_fields()` convenience method
- Foreign key constraint failures when Python SQLite and rlm_core share the same database file
- In-memory store connection lifecycle (persistent connection for `:memory:` mode)
- Evolution log schema compatibility with rlm_core's `node_id`/`reason` column names
- Provenance round-trip fidelity via metadata storage

## [0.6.1] - 2026-02-02

### Fixed
- **hook-dispatch.sh**: Export `CLAUDE_PLUGIN_ROOT` env var so Go binaries can find plugin root (fixes "plugin root not found" error)

### Removed
- `UserPromptSubmit` prompt hook — caused Claude to block normal messages; RLM guidance now lives in CLAUDE.md
- `PostToolUse` catch-all prompt hook — fired on every tool use adding cost/latency; redundant with PreToolUse complexity-check
- `PreCompact` prompt hook — low value; trajectory-save on Stop handles persistence

### Changed
- All hooks are now command-type only (no prompt hooks remain) — eliminates invisible gating behavior
- Hook architecture: command hooks provide dynamic state data, CLAUDE.md provides static workflow instructions

## [0.6.0] - 2026-02-02

### Added
- Go hook binaries replacing Python scripts (~5ms vs ~500ms startup)
- Cross-plugin event system (`~/.claude/events/`) for DP↔RLM coordination
- JSON Schema definitions for all event types
- Python event emission/consumption helpers (`src/events/`)
- Version-aware config migration (V1→V2) preserving user customizations
- Platform-aware hook dispatcher with fallback chain
- GitHub Actions CI for cross-compilation (5 platforms)
- Unit tests for hookio, events, and config packages

### Changed
- **rlm-core enabled by default** — now a required dependency as of v0.7.0
- rlm-core PyO3 bindings now support metadata, provenance, and embedding parameters
- rlm-core uses separate `-core.db` file to avoid schema conflicts with Python SQLite
- hooks.json now uses Go binaries + prompt-based hooks
- Complexity check responds to all DP phases (not just spec/review)
- RLM orchestrator agent informed about parallel tool call behavior
- session-init verifies rlm_core is importable on startup

### Fixed
- Hyperedge queries used `row["type"]` instead of `row["edge_type"]` matching SQLite schema

### Deprecated
- Python hook scripts moved to `scripts/legacy/`
- Set `RLM_USE_LEGACY_HOOKS=1` to use legacy Python hooks
