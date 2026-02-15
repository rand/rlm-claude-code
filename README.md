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

### Clone & Setup

```bash
git clone --recurse-submodules https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code
npm install

# Verify
npm run verify
```

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

| Function | Description |
|----------|-------------|
| `peek(var, start, end)` | View slice of large content |
| `search(var, pattern)` | Find patterns in content |
| `llm(query, context)` | Recursive sub-query |
| `map_reduce(content, map, reduce)` | Parallel processing |
| `memory_query(query)` | Search persistent memory |
| `memory_add_fact(content, conf)` | Store a fact |

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
```

---

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
└─────────────────────────────────────────────┘
    │
    ▼ (if RLM activated)
┌─────────────────────────────────────────────┐
│           RLM EXECUTION ENGINE              │
│  ┌─────────────┐    ┌─────────────────┐    │
│  │ Context Mgr │───►│ REPL Sandbox    │    │
│  │ Externalize │    │ peek/search/llm │    │
│  └─────────────┘    └─────────────────┘    │
│  ┌─────────────┐    ┌─────────────────┐    │
│  │ Recursive   │    │ Tool Bridge     │    │
│  │ Handler     │    │ bash/read/grep  │    │
│  └─────────────┘    └─────────────────┘    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│            PERSISTENCE LAYER                │
│  Memory Store ─────► Memory Evolution       │
│  Reasoning Traces ──► Strategy Cache        │
└─────────────────────────────────────────────┘
```

---

## Common Commands

| Command | Description |
|---------|-------------|
| `npm install` | Full setup (downloads pre-built binaries) |
| `npm run ensure-setup` | Check/fix missing dependencies |
| `npm run verify` | Verify installation |
| `npm run test` | Run smoke tests |
| `npm run test:full` | Run full test suite |
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

---

## License

MIT License - See [LICENSE](LICENSE) for details.
