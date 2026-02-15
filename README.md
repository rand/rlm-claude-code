# RLM-Claude-Code

Transform Claude Code into a Recursive Language Model (RLM) agent with intelligent orchestration, unbounded context handling, persistent memory, and REPL-based decomposition.

**rlm-core integration**: This project bundles [rlm-core](https://github.com/rand/loop) as a required dependency, providing shared Rust-based implementations with [recurse](https://github.com/rand/recurse). rlm-core provides 10-50x faster pattern classification via PyO3 bindings. Pre-built wheels are available for common platforms.

## What is RLM?

RLM (Recursive Language Model) enables Claude to handle arbitrarily large contexts by decomposing complex tasks into smaller sub-queries. Instead of processing 500K tokens at once, RLM lets Claude:

- **Peek** at context structure before deep analysis
- **Search** using patterns to narrow focus
- **Partition** large contexts and process in parallel via map-reduce
- **Recurse** with sub-queries for verification
- **Remember** facts and experiences across sessions

This results in better accuracy on complex tasks while optimizing cost through intelligent model selection.

---

## Quick Start

### Prerequisites

- **Python 3.12+** — `brew install python@3.12` or [python.org](https://python.org)
- **uv** package manager — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js 20+** — `brew install node` or [nodejs.org](https://nodejs.org)

### Clone & Setup (Pre-built Binaries)

```bash
git clone --recurse-submodules https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code
npm install

# Verify
npm run verify
```

### Build from Source (Requires Rust)

If you want to build rlm-core from source instead of using pre-built wheels:

```bash
# Prerequisites
# - Rust 1.75+: rustup update stable
# - maturin: pip install maturin

git clone --recurse-submodules https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code

# Build rlm-core Rust library
maturin develop --release

# Install Python dependencies
uv sync --all-extras

# Verify
python -c "import rlm_core; print(rlm_core.version())"
uv run pytest tests/ -v
```

**Note**: The `python-source` in `pyproject.toml` should point to `"vendor/loop/rlm-core/python"` for maturin to find the Python module correctly.

### Install as Claude Code Plugin

```bash
claude plugin install . --scope user
```

Or install from marketplace:
```bash
claude plugin marketplace add github:rand/rlm-claude-code
claude plugin install rlm-claude-code@rlm-claude-code
```

---

## Self-Healing Setup

The plugin automatically checks and fixes dependencies on first use:

- Checks: `uv`, Python venv, Go binaries, `rlm_core` wheel
- Downloads missing components from GitHub releases
- AI assistant sees status and can help if issues found

```bash
# Manual check
npm run ensure-setup

# Auto-fix
npm run ensure-setup -- --fix
```

---

## Usage

### Slash Commands

| Command | Description |
|---------|-------------|
| `/rlm-claude-code:rlm` | Show RLM status |
| `/rlm-claude-code:rlm activate` | Launch RLM orchestrator |
| `/rlm-claude-code:rlm on` | Enable RLM mode |
| `/rlm-claude-code:rlm off` | Disable RLM mode |
| `/rlm-claude-code:rlm-orchestrator` | Complex context tasks |
| `/rlm-claude-code:simple` | Bypass RLM for one query |

### REPL Helper Functions

RLM provides a sandboxed Python environment with helper functions:

**Context Variables:**
- `conversation` — List of message dicts with role and content
- `files` — Dict mapping filenames to content
- `tool_outputs` — List of tool execution results
- `working_memory` — Scratchpad for intermediate results

**Helper Functions:**

| Function | Description |
|----------|-------------|
| `peek(var, start, end)` | View slice of large content |
| `search(var, pattern, regex=False)` | Find patterns in content |
| `summarize(var, max_tokens)` | LLM-powered summarization |
| `llm(query, context, spawn_repl)` | Recursive sub-query |
| `llm_batch([(q1,c1), (q2,c2), ...])` | Parallel LLM calls |
| `map_reduce(content, map, reduce, n_chunks)` | Partition and aggregate |
| `find_relevant(content, query, top_k)` | Find most relevant sections |

**Memory Functions** (when enabled):

| Function | Description |
|----------|-------------|
| `memory_query(query, limit)` | Search stored knowledge |
| `memory_add_fact(content, confidence)` | Store a fact |
| `memory_add_experience(content, outcome, success)` | Store an experience |
| `memory_get_context(limit)` | Retrieve relevant context |
| `memory_relate(node1, node2, relation)` | Create relationships |

### Example

```python
# In RLM REPL environment:
conversation  # List of messages
files         # Dict of filename → content

# View slice of a file
peek(files['main.py'], 0, 500)

# Search for patterns
search(files, 'def authenticate')

# Recursive sub-query
llm("Summarize this function", context=files['auth.py'])

# Parallel processing
map_reduce(large_content, "Extract key points", "Combine into summary", n_chunks=4)
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              INTELLIGENT ORCHESTRATOR                   │
│  ┌───────────────────┐   ┌───────────────────────────┐  │
│  │ Complexity        │   │ Orchestration Decision    │  │
│  │ Classifier        │   │ • Activate RLM?           │  │
│  │ • Token count     │──►│ • Which model tier?       │  │
│  │ • Cross-file refs │   │ • Depth budget (0-3)?     │  │
│  │ • Query patterns  │   │ • Tool access level?      │  │
│  └───────────────────┘   └───────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
    │
    ▼ (if RLM activated)
┌─────────────────────────────────────────────────────────┐
│                 RLM EXECUTION ENGINE                    │
│                                                         │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │  Context Manager │    │     REPL Sandbox         │   │
│  │  • Externalize   │───►│  • peek(), search()      │   │
│  │    conversation  │    │  • llm(), llm_batch()    │   │
│  │  • files, tools  │    │  • map_reduce()          │   │
│  └──────────────────┘    │  • memory_*() functions  │   │
│                          └──────────────────────────┘   │
│                                     │                   │
│                                     ▼                   │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ Recursive Handler│    │    Tool Bridge           │   │
│  │ • Depth ≤ 3      │    │  • bash, read, grep      │   │
│  │ • Model cascade  │    │  • Permission control    │   │
│  │ • Sub-query spawn│    │  • Blocked commands      │   │
│  └──────────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  PERSISTENCE LAYER                      │
│                                                         │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │  Memory Store    │    │   Reasoning Traces       │   │
│  │  • Facts, exps   │    │  • Goals, decisions      │   │
│  │  • Hyperedges    │    │  • Options, outcomes     │   │
│  │  • SQLite + WAL  │    │  • Decision trees        │   │
│  └──────────────────┘    └──────────────────────────┘   │
│           │                         │                   │
│           ▼                         ▼                   │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ Memory Evolution │    │   Strategy Cache         │   │
│  │ task → session   │    │  • Learn from success    │   │
│  │ session → long   │    │  • Similarity matching   │   │
│  │ decay → archive  │    │  • Suggest strategies    │   │
│  └──────────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Final Answer
```

---

## Core Components

### Memory System

Persistent storage for cross-session learning:

- **Node Types**: facts, experiences, procedures, goals
- **Memory Tiers**: task → session → longterm → archive
- **Session Isolation**: task/session tiers are isolated per terminal
- **Hyperedges**: N-ary relationships with typed roles
- **Storage**: SQLite with WAL mode for concurrent access

```python
from src import MemoryStore, MemoryEvolution

# Create and use memory
store = MemoryStore(db_path="~/.claude/rlm-memory.db")
fact_id = store.create_node(
    node_type="fact",
    content="This project uses FastAPI",
    confidence=0.9,
)

# Evolve memory through tiers
evolution = MemoryEvolution(store)
evolution.consolidate(task_id="current-task")  # task → session
evolution.promote(session_id="current-session")  # session → longterm
evolution.decay(days_threshold=30)  # longterm → archive
```

### Reasoning Traces

Track decision-making for transparency and debugging:

```python
from src import ReasoningTraces

traces = ReasoningTraces(store)

# Create goal and decision tree
goal_id = traces.create_goal("Implement user authentication")
decision_id = traces.create_decision(goal_id, "Choose auth strategy")

# Track options considered
jwt_option = traces.add_option(decision_id, "Use JWT tokens")
session_option = traces.add_option(decision_id, "Use session cookies")

# Record choice and reasoning
traces.choose_option(decision_id, jwt_option)
traces.reject_option(decision_id, session_option, "JWT better for API")

# Get full decision tree
tree = traces.get_decision_tree(goal_id)
```

---

## Configuration

RLM stores configuration at `~/.claude/rlm-config.json`:

```json
{
  "activation": {
    "mode": "complexity",
    "fallback_token_threshold": 80000
  },
  "depth": {
    "default": 2,
    "max": 3
  },
  "trajectory": {
    "verbosity": "normal",
    "streaming": true
  }
}
```

| Key | Type | Description |
|-----|------|-------------|
| `activation.mode` | string | `"complexity"` (default), `"always"`, `"never"` |
| `depth.default` | int | Default recursion depth (1-3) |
| `depth.max` | int | Maximum recursion depth (default: 3) |
| `trajectory.verbosity` | string | `"minimal"`, `"normal"`, `"verbose"`, `"debug"` |

---

## Hooks

### Go Binary Hooks (v0.6.0+)

RLM uses compiled Go binaries for hook execution, reducing startup latency from ~500ms (Python) to ~5ms:

| Hook | Binary | Purpose |
|------|--------|---------|
| `SessionStart` | `session-init` | Initialize RLM environment |
| `UserPromptSubmit` | `complexity-check` | Decide if RLM should activate |
| `Stop` | `trajectory-save` | Save trajectory on session end |

### Event System

Hooks emit and consume events via `~/.claude/events/`, enabling coordination between plugins.

---

## Common Commands

| Command | Description |
|---------|-------------|
| `npm install` | Full setup (downloads pre-built binaries) |
| `npm run ensure-setup` | Check/fix missing dependencies |
| `npm run verify` | Verify installation |
| `npm run test` | Run smoke tests |
| `npm run test:full` | Run full test suite (3000+ tests) |
| `npm run build -- --all` | Build from source (requires Rust + Go) |

---

## Building from Source

For advanced users who want to build Rust/Go components:

```bash
# Requires Rust 1.75+ and Go 1.21+
npm run build -- --all

# Or build specific components:
npm run build -- --binaries-only   # Go hooks only
npm run build -- --wheel-only      # rlm-core wheel only
```

### Building rlm-core (Rust Library)

If you need to build or modify the rlm-core Rust library:

```bash
# From project root
cd vendor/loop/rlm-core

# Build and install for development
maturin develop --release

# Or build a wheel
maturin build --release
```

The built wheel will be in `target/wheels/`.

---

## Troubleshooting

### Setup Issues

```bash
# Check what's missing
npm run ensure-setup

# Auto-fix
npm run ensure-setup -- --fix
```

### Tests Failing

```bash
uv sync --all-extras
uv run pytest tests/ -v --tb=long
```

### RLM Not Initializing

```bash
# Verify plugin is installed
claude plugins list

# Check hooks exist
ls hooks/hooks.json
```

### Reset Everything

```bash
rm ~/.claude/rlm-config.json
rm ~/.claude/rlm-memory.db
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Step-by-step installation |
| [User Guide](docs/user-guide.md) | Complete usage documentation |
| [Architecture](docs/process/architecture.md) | Design decisions (ADRs) |
| [SPEC Overview](docs/spec/00-overview.md) | Capability specifications |

---

## Development

See [CLAUDE.md](CLAUDE.md) for development instructions.

```bash
# Type check
uv run ty check src/

# Run all tests
npm run test:full

# Run npm script tests
npm run test:npm
```

### Running Tests

```bash
# Quick iteration (fast mode)
HYPOTHESIS_PROFILE=fast uv run pytest tests/unit/ -v

# Standard development
uv run pytest tests/ -v

# Full CI run
HYPOTHESIS_PROFILE=ci uv run pytest tests/ -v
```

---

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601v1) - Zhang, Kraska, Khattab
- [Claude Code Plugins](https://docs.anthropic.com/en/docs/claude-code)

---

## License

MIT License - See [LICENSE](LICENSE) for details.
