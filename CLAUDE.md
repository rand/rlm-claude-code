# RLM-Claude-Code

Transform Claude Code into a Recursive Language Model agent with intelligent orchestration, persistent memory, and REPL-based context decomposition.

## Quick Start

```bash
# Setup
uv sync --all-extras

# Type check
uv run ty check src/

# Test (1000+ tests)
uv run pytest tests/ -v

# Install as plugin
claude plugins install . --scope user
```

## Project Structure

```
rlm-claude-code/
├── CLAUDE.md                       # You are here
├── README.md                       # User-facing documentation
├── rlm-claude-code-spec.md         # Technical specification
├── docs/
│   ├── getting-started.md          # Installation guide
│   ├── user-guide.md               # Complete usage docs
│   ├── spec/                       # Capability specifications
│   │   ├── 00-overview.md          # SPEC index
│   │   ├── 01-repl-functions.md    # Advanced REPL (SPEC-01)
│   │   ├── 02-memory-foundation.md # Memory store (SPEC-02)
│   │   ├── 03-memory-evolution.md  # Memory tiers (SPEC-03)
│   │   ├── 04-reasoning-traces.md  # Decision tracking (SPEC-04)
│   │   └── 05-budget-tracking.md   # Cost control (SPEC-05)
│   └── process/
│       ├── README.md               # Process index
│       ├── architecture.md         # ADRs
│       ├── implementation.md       # Implementation phases
│       ├── code-review.md          # Review checklist
│       ├── testing.md              # Testing strategy
│       └── debugging.md            # Debug workflow
├── src/
│   ├── __init__.py                 # Public API exports
│   ├── types.py                    # Core types
│   ├── config.py                   # Configuration
│   ├── orchestrator.py             # Main RLM loop
│   ├── intelligent_orchestrator.py # Claude-powered decisions
│   ├── auto_activation.py          # Complexity-based activation
│   ├── context_manager.py          # Context externalization
│   ├── repl_environment.py         # Sandboxed Python REPL
│   ├── recursive_handler.py        # Sub-query management
│   ├── memory_store.py             # SQLite memory (SPEC-02)
│   ├── memory_evolution.py         # Memory tiers (SPEC-03)
│   ├── reasoning_traces.py         # Decision trees (SPEC-04)
│   ├── enhanced_budget.py          # Cost tracking (SPEC-05)
│   ├── cost_tracker.py             # Token/cost accounting
│   ├── trajectory.py               # Event logging
│   ├── trajectory_analysis.py      # Strategy extraction
│   ├── strategy_cache.py           # Learn from success
│   ├── tool_bridge.py              # Controlled tool access
│   ├── api_client.py               # LLM API wrapper
│   ├── smart_router.py             # Model selection
│   └── ...
├── tests/
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── property/                   # Hypothesis tests
│   ├── security/                   # Security tests
│   └── benchmarks/                 # Performance tests
├── scripts/                        # Hook scripts
├── hooks/                          # hooks.json
└── commands/                       # Slash commands
```

## Essential Context

**Read before making changes:**

1. `README.md` — Architecture and component overview
2. `docs/spec/00-overview.md` — Capability specifications
3. `docs/process/architecture.md` — Design decisions (ADRs)

## Development Commands

```bash
# Type check (must pass)
uv run ty check src/

# Lint (must pass)
uv run ruff check src/ --fix
uv run ruff format src/

# All tests
uv run pytest tests/ -v

# By category
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run pytest tests/property/ -v -m hypothesis
uv run pytest tests/security/ -v
uv run pytest tests/benchmarks/ --benchmark-only

# With coverage
uv run pytest tests/ -v --cov=src/ --cov-report=html
```

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│         INTELLIGENT ORCHESTRATOR            │
│  • Complexity classification                │
│  • Model selection (Opus/Sonnet/Haiku)     │
│  • Depth budget (0-3)                       │
│  • Tool access level                        │
└─────────────────────────────────────────────┘
    │
    ▼ (if RLM activated)
┌─────────────────────────────────────────────┐
│           RLM EXECUTION ENGINE              │
│  ┌─────────────┐    ┌─────────────────┐    │
│  │ Context Mgr │───►│ REPL Sandbox    │    │
│  │ Externalize │    │ peek/search/llm │    │
│  └─────────────┘    │ map_reduce      │    │
│                     │ memory_*        │    │
│  ┌─────────────┐    └─────────────────┘    │
│  │ Recursive   │    ┌─────────────────┐    │
│  │ Handler     │    │ Tool Bridge     │    │
│  │ depth ≤ 3   │    │ bash/read/grep  │    │
│  └─────────────┘    └─────────────────┘    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│            PERSISTENCE LAYER                │
│  Memory Store ─────► Memory Evolution       │
│  (SQLite+WAL)        (task→session→long)   │
│  Reasoning Traces ──► Strategy Cache        │
│  (decision trees)     (learn from success)  │
└─────────────────────────────────────────────┘
    │
    ▼
Budget Tracking → Trajectory → Final Answer
```

## Implementation Status

All phases complete:

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core Infrastructure | Complete |
| 2 | Claude Code Integration | Complete |
| 3 | Optimization | Complete |
| 4 | Advanced Features | Complete |
| 5 | Intelligent Orchestration | Complete |

SPEC implementations:

| Spec | Component | Status |
|------|-----------|--------|
| SPEC-01 | Advanced REPL Functions | Complete |
| SPEC-02 | Memory Foundation | Complete |
| SPEC-03 | Memory Evolution | Complete |
| SPEC-04 | Reasoning Traces | Complete |
| SPEC-05 | Enhanced Budget Tracking | Complete |

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

## REPL Helper Functions

| Function | Purpose |
|----------|---------|
| `peek(var, start, end)` | View slice of context |
| `search(var, pattern, regex)` | Find patterns |
| `summarize(var, max_tokens)` | LLM summarization |
| `llm(query, context)` | Recursive sub-query |
| `llm_batch(queries)` | Parallel sub-queries |
| `map_reduce(content, map_p, reduce_p)` | Partition+aggregate |
| `find_relevant(content, query, top_k)` | Relevance search |
| `extract_functions(content)` | Parse functions |
| `memory_query(query)` | Search memory |
| `memory_add_fact(content, conf)` | Store fact |
| `memory_add_experience(...)` | Store experience |

## Code Style

- Type annotations on all public functions
- Google-style docstrings with spec references
- No functions >50 lines
- Pydantic models at API boundaries

## Testing Requirements

- Unit tests for all functions
- Property tests for data transformations
- Security tests for REPL operations
- Integration tests for component interactions
- 1000+ tests total

## Before Committing

1. `uv run ty check src/` — Must pass
2. `uv run ruff check src/` — Must pass
3. `uv run pytest tests/ -v` — Must pass
4. Run `/code-review` if significant changes

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/rlm` | Toggle/configure RLM mode |
| `/rlm status` | Show configuration |
| `/rlm mode <fast\|balanced\|thorough>` | Set mode |
| `/simple` | Bypass RLM once |
| `/trajectory <file>` | Analyze trajectory |
| `/test` | Run tests |
| `/bench` | Run benchmarks |
| `/code-review` | Review changes |

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601v1)
- [README](./README.md) — Full documentation
- [User Guide](./docs/user-guide.md) — Usage details
