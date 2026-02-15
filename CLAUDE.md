# RLM-Claude-Code

Context for developers working on the RLM-Claude-Code project.

## Quick Start

```bash
# Setup (downloads pre-built binaries)
npm install

# Verify installation
npm run verify

# Type check
uv run ty check src/

# Test (3000+ tests)
npm run test:full

# Install as plugin
claude plugins install . --scope user
```

## Project Structure

```
rlm-claude-code/
├── CLAUDE.md                       # You are here (developer context)
├── README.md                       # User-facing overview
├── docs/
│   ├── getting-started.md          # Installation guide
│   ├── user-guide.md               # Complete usage docs
│   ├── spec/                       # Capability specifications
│   └── process/                    # Architecture, ADRs, testing
├── src/                            # Python source code
│   ├── orchestrator.py             # Main RLM loop
│   ├── intelligent_orchestrator.py # Claude-powered decisions
│   ├── context_manager.py          # Context externalization
│   ├── repl_environment.py         # Sandboxed Python REPL
│   ├── memory_store.py             # SQLite memory (SPEC-02)
│   ├── memory_evolution.py         # Memory tiers (SPEC-03)
│   ├── reasoning_traces.py         # Decision trees (SPEC-04)
│   ├── enhanced_budget.py          # Cost tracking (SPEC-05)
│   └── ...
├── tests/                          # Test suite
├── scripts/npm/                    # TypeScript npm scripts
│   ├── ensure-setup.ts             # Self-healing setup
│   ├── hook-dispatch.ts            # Cross-platform hook dispatcher
│   ├── download-binaries.ts        # Download Go binaries
│   └── download-wheel.ts           # Download Python wheel
├── hooks/                          # hooks.json (Claude Code hooks)
└── commands/                       # Slash commands
```

## Essential Context

**Read before making changes:**

1. `README.md` — Architecture overview
2. `docs/spec/00-overview.md` — Capability specifications
3. `docs/process/architecture.md` — Design decisions (ADRs)

## Development Commands

```bash
# Setup
npm install           # Full setup with pre-built binaries
npm run ensure-setup  # Check/fix missing dependencies
npm run verify        # Verify installation

# Testing
npm run test          # Run smoke tests
npm run test:full     # Run full test suite (3000+ tests)
npm run test:npm      # Run TypeScript tests for npm scripts

# Building
npm run build         # Download binaries + install deps
npm run build -- --all  # Build from source (needs Rust + Go)

# Python tools
uv run ty check src/  # Type check (must pass)
uv run ruff check src/ --fix  # Lint (must pass)
uv run ruff format src/       # Format

# Direct pytest
uv run pytest tests/ -v       # All tests
uv run pytest tests/unit/ -v  # Unit tests only
```

## Key Technologies

| Tool | Purpose |
|------|---------|
| uv | Package management |
| ty | Type checking |
| ruff | Linting/formatting |
| pydantic | Data validation |
| hypothesis | Property testing |
| RestrictedPython | REPL sandbox |
| SQLite | Memory persistence |

## Code Style

- Type annotations on all public functions
- Google-style docstrings with spec references
- No functions >50 lines
- Pydantic models at API boundaries

## Before Committing

1. `uv run ty check src/` — Must pass
2. `uv run ruff check src/` — Must pass
3. `uv run pytest tests/ -v` — Must pass
4. `npm run test:npm` — Must pass (npm scripts)

## Contributing

### Commit Message Format

Use conventional commits:

```
<type>(<scope>): <description>

[optional body]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### PR Checklist

- [ ] Type check passes: `uv run ty check src/`
- [ ] Lint passes: `uv run ruff check src/`
- [ ] Tests pass: `uv run pytest tests/ -v`
- [ ] NPM tests pass: `npm run test:npm`
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventional format

## Self-Healing Setup System

The plugin includes automatic dependency management:

### Components

| File | Purpose |
|------|---------|
| `scripts/npm/ensure-setup.ts` | Checks and fixes dependencies |
| `scripts/npm/hook-dispatch.ts` | Cross-platform hook dispatcher |
| `scripts/npm/download-binaries.ts` | Downloads Go binaries from GitHub |
| `scripts/npm/download-wheel.ts` | Downloads Python wheel from GitHub |
| `hooks/hooks.json` | Hook configuration (uses npx ts-node) |

### How It Works

1. On SessionStart, `ensure-setup.ts --json` runs
2. Checks: `uv`, venv, binaries, `rlm_core`
3. Outputs JSON status to AI via `hookSpecificOutput`
4. AI can guide user to fix issues if `needsAttention: true`

### Installation Modes

| Mode | Detection | Behavior |
|------|-----------|----------|
| marketplace | No `.git`, no `/dev/` in path | Download from GitHub releases |
| dev | `.git` exists or `/dev/` in path | Prompt to build from source |

## Implementation Status

| Spec | Component | Status |
|------|-----------|--------|
| SPEC-01 | Advanced REPL Functions | Complete |
| SPEC-02 | Memory Foundation | Complete |
| SPEC-03 | Memory Evolution | Complete |
| SPEC-04 | Reasoning Traces | Complete |
| SPEC-05 | Enhanced Budget Tracking | Complete |

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601v1)
- [README](./README.md) — User overview
- [Getting Started](./docs/getting-started.md) — Installation
- [User Guide](./docs/user-guide.md) — Usage details
- [SPEC Overview](./docs/spec/00-overview.md) — Specifications
