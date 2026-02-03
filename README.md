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

- **Python 3.12+**: `brew install python@3.12` or [python.org](https://python.org)
- **uv** (Python package manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Rust 1.75+** (for dev builds): `rustup update stable` — not needed if installing from pre-built wheels

### Installation (from pre-built wheel)

Download the wheel for your platform from [GitHub Releases](https://github.com/rand/rlm-claude-code/releases), then:

```bash
pip install rlm_claude_code-*.whl
```

### Installation (from source, requires Rust)

```bash
# Clone with submodules (rlm-core is bundled)
git clone --recurse-submodules https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code

# Build and install (compiles rlm-core Rust library)
maturin develop --release

# Install Python dependencies
uv sync --all-extras

# Verify
python -c "import rlm_core; print(rlm_core.version())"
uv run pytest tests/ -v
```

### As a Claude Code Plugin

#### Step 1: Install the Plugin

```bash
# Add the marketplace (one-time setup)
claude plugin marketplace add github:rand/rlm-claude-code

# Install the plugin
claude plugin install rlm-claude-code@rlm-claude-code
```

#### Step 2: Set Up the Plugin Environment

The plugin includes rlm-core (bundled via maturin). After installation:

```bash
# Navigate to the plugin directory (adjust version number as needed)
PLUGIN_DIR=~/.claude/plugins/cache/rlm-claude-code/rlm-claude-code/$(ls ~/.claude/plugins/cache/rlm-claude-code/rlm-claude-code/ | sort -V | tail -1)
cd "$PLUGIN_DIR"

# Create venv and install (includes rlm-core)
uv venv && uv sync

# Verify rlm-core is available
"$PLUGIN_DIR/.venv/bin/python" -c "import rlm_core; print('rlm_core OK:', rlm_core.version())"
```

#### Step 4: Verify Installation

Restart Claude Code. You should see "RLM initialized" on startup.

Test the hooks manually:
```bash
PLUGIN_DIR=~/.claude/plugins/cache/rlm-claude-code/rlm-claude-code/$(ls ~/.claude/plugins/cache/rlm-claude-code/rlm-claude-code/ | sort -V | tail -1)
"$PLUGIN_DIR/.venv/bin/python" "$PLUGIN_DIR/scripts/init_rlm.py"
# Should print: RLM initialized
```

#### Updating the Plugin

When updating, you'll need to re-create the venv:

```bash
claude plugin update rlm-claude-code@rlm-claude-code

# Re-setup the new version
PLUGIN_DIR=~/.claude/plugins/cache/rlm-claude-code/rlm-claude-code/$(ls ~/.claude/plugins/cache/rlm-claude-code/rlm-claude-code/ | sort -V | tail -1)
cd "$PLUGIN_DIR" && uv venv && uv sync

# rlm-core is bundled — no extra linking needed
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
│  └──────────────────┘    │  • find_relevant()       │   │
│                          │  • memory_*() functions  │   │
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
┌─────────────────────────────────────────────────────────┐
│                  BUDGET & TRAJECTORY                    │
│  • Token tracking per component                         │
│  • Cost limits with alerts                              │
│  • Streaming trajectory output                          │
│  • JSON export for analysis                             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Final Answer
```

---

## rlm-core Integration

RLM-Claude-Code bundles [rlm-core](https://github.com/rand/loop), a Rust library providing high-performance pattern classification, memory storage, and trajectory tracking via PyO3 bindings.

rlm-core is a required dependency — there is no Python fallback mode.

### Configuration

| Setting | Behavior |
|---------|----------|
| `activation.mode` | `"complexity"` (default), `"always"`, `"never"` |
| `depth.default` | Default recursion depth (default: 2) |
| `depth.max` | Maximum recursion depth (default: 3) |

### Performance

| Component | Implementation |
|-----------|---------------|
| Pattern Classifier | Rust via PyO3 |
| Trajectory Events | Rust types via PyO3 |
| Memory Store | rlm_core SQLite (nodes/edges) + Python SQLite (auxiliary tables) |

### Benefits

- **Consistency**: Same classification logic as recurse TUI
- **Performance**: Rust pattern matching is significantly faster
- **Shared Codebase**: Bug fixes benefit both projects

### Python Bindings Usage

```python
from src.complexity_classifier import should_activate_rlm, extract_complexity_signals
from src.types import SessionContext, Message, MessageRole

# Delegates to rlm_core.PatternClassifier
ctx = SessionContext(messages=[Message(role=MessageRole.USER, content="test")])
signals = extract_complexity_signals("Find security vulnerabilities", ctx)
activate, reason = should_activate_rlm("Find security vulnerabilities", ctx)
```

---

## Core Components

### REPL Environment

The REPL provides a sandboxed Python environment for context manipulation:

**Context Variables:**
- `conversation` - List of message dicts with role and content
- `files` - Dict mapping filenames to content
- `tool_outputs` - List of tool execution results
- `working_memory` - Scratchpad for intermediate results

**Helper Functions:**

| Function | Description |
|----------|-------------|
| `peek(var, start, end)` | View a slice of any context variable |
| `search(var, pattern, regex=False)` | Find patterns in context |
| `summarize(var, max_tokens)` | LLM-powered summarization |
| `llm(query, context, spawn_repl)` | Spawn recursive sub-query |
| `llm_batch([(q1,c1), (q2,c2), ...])` | Parallel LLM calls |
| `map_reduce(content, map_prompt, reduce_prompt, n_chunks)` | Partition and aggregate |
| `find_relevant(content, query, top_k)` | Find most relevant sections |
| `extract_functions(content)` | Parse function definitions |
| `run_tool(tool, *args)` | Safe subprocess (uv, ty, ruff) |

**Memory Functions** (when enabled):

| Function | Description |
|----------|-------------|
| `memory_query(query, limit)` | Search stored knowledge |
| `memory_add_fact(content, confidence)` | Store a fact |
| `memory_add_experience(content, outcome, success)` | Store an experience |
| `memory_get_context(limit)` | Retrieve relevant context |
| `memory_relate(node1, node2, relation)` | Create relationships |

**Available Libraries:**

| Library | Alias | Description |
|---------|-------|-------------|
| `re` | - | Regular expressions |
| `json` | - | JSON encoding/decoding |
| `pydantic` | `BaseModel`, `Field` | Data validation |
| `hypothesis` | `given`, `st` | Property-based testing |
| `cpmpy` | `cp` | Constraint programming |
| `numpy` | `np` | Numerical computing |
| `pandas` | `pd` | DataFrames and analysis |
| `polars` | `pl` | Fast DataFrames |
| `seaborn` | `sns` | Statistical visualization |

### Memory System

Persistent storage for cross-session learning:

- **Node Types**: facts, experiences, procedures, goals
- **Memory Tiers**: task → session → longterm → archive
- **Session Isolation**: task/session tiers are isolated per terminal; longterm/archive shared (see [docs/session-isolation.md](docs/session-isolation.md))
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

### Budget Tracking

Granular cost control with configurable limits:

```python
from src import EnhancedBudgetTracker, BudgetLimits

tracker = EnhancedBudgetTracker()
tracker.set_limits(BudgetLimits(
    max_cost_per_task=5.0,
    max_recursive_calls=10,
    max_depth=3,
))

# Check before expensive operations
allowed, reason = tracker.can_make_llm_call()
if not allowed:
    print(f"Budget exceeded: {reason}")

# Get detailed metrics
metrics = tracker.get_metrics()
print(f"Cost: ${metrics.total_cost_usd:.2f}")
print(f"Depth: {metrics.max_depth_reached}")
print(f"Calls: {metrics.sub_call_count}")
```

---

## Using RLM

### Slash Commands

| Command | Description |
|---------|-------------|
| `/rlm` | Show current status |
| `/rlm on` | Enable RLM for this session |
| `/rlm off` | Disable RLM mode |
| `/rlm status` | Full configuration display |
| `/rlm mode <fast\|balanced\|thorough>` | Set execution mode |
| `/rlm depth <0-3>` | Set max recursion depth |
| `/rlm budget $X` | Set session cost limit |
| `/rlm model <opus\|sonnet\|haiku\|auto>` | Force model selection |
| `/rlm tools <none\|repl\|read\|full>` | Set sub-LLM tool access |
| `/rlm verbosity <minimal\|normal\|verbose\|debug>` | Set output detail |
| `/rlm reset` | Reset to defaults |
| `/rlm save` | Save preferences to disk |
| `/simple` | Bypass RLM for current query |
| `/trajectory <file>` | Analyze a trajectory file |
| `/test` | Run test suite |
| `/bench` | Run benchmarks |
| `/code-review` | Review code changes |

### Execution Modes

| Mode | Depth | Model | Tools | Best For |
|------|-------|-------|-------|----------|
| `fast` | 1 | Haiku | REPL only | Quick questions, iteration |
| `balanced` | 2 | Sonnet | Read-only | Most tasks (default) |
| `thorough` | 3 | Opus | Full access | Complex debugging, architecture |

### Auto-Activation

RLM automatically activates when it detects:
- **Large context**: >80K tokens in conversation
- **Cross-file reasoning**: Questions spanning multiple files
- **Complex debugging**: Stack traces, error analysis
- **Architecture questions**: System design, refactoring patterns

Force activation with `/rlm on` or bypass with `/simple`.

---

## Configuration

### Config File

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
| `activation.mode` | string | `"micro"`, `"complexity"`, `"always"`, `"manual"` |
| `depth.default` | int | Default recursion depth (1-3) |
| `trajectory.verbosity` | string | `"minimal"`, `"normal"`, `"verbose"`, `"debug"` |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API access (uses Claude Code's key) |
| `OPENAI_API_KEY` | OpenAI API access (optional, for GPT models) |
| `RLM_CONFIG_PATH` | Custom config file location |
| `RLM_DEBUG` | Enable debug logging |

---

## Hooks

### Go Binary Hooks (v0.6.0+)

As of v0.6.0, RLM uses compiled Go binaries for hook execution, reducing startup latency from ~500ms (Python) to ~5ms. The three primary hooks are:

| Hook | Binary | Purpose |
|------|--------|---------|
| `SessionStart` | `session-init` | Initialize RLM environment, emit `session.started` event |
| `UserPromptSubmit` | `complexity-check` | Decide if RLM should activate, responds to DP phases |
| `Stop` | `trajectory-save` | Save trajectory and emit `session.ended` event |

Prompt-based hooks (no binary needed) handle context sync, output capture, and pre-compaction.

#### Cross-Plugin Event System

Hooks emit and consume events via `~/.claude/events/`, enabling coordination between plugins (e.g., DP and RLM). All event types have JSON Schema definitions. Python helpers in `src/events/` provide emit/consume APIs.

#### Config Migration

v0.6.0 introduces a V2 config format. Existing V1 configs are automatically migrated on first run, preserving user customizations.

#### Legacy Fallback

To use the original Python hook scripts:

```bash
export RLM_USE_LEGACY_HOOKS=1
```

Legacy scripts are located in `scripts/legacy/`.

### Legacy Hook Reference

The original Python hooks (now in `scripts/legacy/`):

| Hook | Script | Purpose |
|------|--------|---------|
| `SessionStart` | `init_rlm.py` | Initialize RLM environment |
| `UserPromptSubmit` | `check_complexity.py` | Decide if RLM should activate |
| `PreToolUse` | `sync_context.py` | Sync tool context with RLM state |
| `PostToolUse` | `capture_output.py` | Capture tool output for context |
| `PreCompact` | `externalize_context.py` | Externalize before compaction |
| `Stop` | `save_trajectory.py` | Save trajectory on session end |

### Hook Setup (Important!)

Claude Code plugins register hooks in `hooks/hooks.json`, but these need to be merged into your global `~/.claude/settings.json` to take effect.

**After installing or updating this plugin, run:**

```bash
python3 ~/.claude/scripts/merge-plugin-hooks.py
```

If you don't have the merge script, create it:

```bash
mkdir -p ~/.claude/scripts
curl -o ~/.claude/scripts/merge-plugin-hooks.py \
  https://raw.githubusercontent.com/rand/rlm-claude-code/main/scripts/merge-plugin-hooks.py
```

### Hook Path Design

This plugin uses `${CLAUDE_PLUGIN_ROOT}` variable and `.venv/bin/python` (not `uv run`) to ensure hooks work correctly:

- **`${CLAUDE_PLUGIN_ROOT}`** - Expands to the plugin's install path, ensuring hooks work after updates
- **`.venv/bin/python`** - Direct venv Python, avoiding `uv run` sandbox issues on macOS

### Verifying Hooks

Check hooks are correctly registered:

```bash
# See which hooks are active
cat ~/.claude/settings.json | jq '.hooks.SessionStart'

# Verify paths point to current version
cat ~/.claude/plugins/installed_plugins.json | jq '.plugins["rlm-claude-code@rlm-claude-code"][0].version'
```

If hook paths don't match the installed version, re-run the merge script.

### Troubleshooting Hooks

**"RLM not initializing"**
1. Run `python3 ~/.claude/scripts/merge-plugin-hooks.py`
2. Restart Claude Code
3. Check: `cat ~/.claude/settings.json | jq '.hooks.SessionStart'`

**Hooks pointing to old version**
- This happens after plugin updates
- Run `python3 ~/.claude/scripts/merge-plugin-hooks.py` after each update

**macOS sandbox errors with uv**
- This plugin already uses `.venv/bin/python` to avoid this issue
- If you see `SCDynamicStore` errors, verify hooks don't use `uv run`

---

## Development

### Setup

```bash
git clone https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code

# Install all dependencies
uv sync --all-extras

# Run tests (3000+ tests)
uv run pytest tests/ -v

# Type check
uv run ty check src/

# Lint and format
uv run ruff check src/ --fix
uv run ruff format src/
```

### Project Structure

```
rlm-claude-code/
├── src/
│   ├── orchestrator.py           # Main RLM loop
│   ├── intelligent_orchestrator.py  # Claude-powered decisions
│   ├── auto_activation.py        # Complexity-based activation
│   ├── context_manager.py        # Context externalization
│   ├── repl_environment.py       # Sandboxed Python REPL
│   ├── recursive_handler.py      # Sub-query management
│   ├── memory_store.py           # Persistent memory (SQLite)
│   ├── memory_evolution.py       # Memory tier management
│   ├── reasoning_traces.py       # Decision tree tracking
│   ├── enhanced_budget.py        # Cost tracking and limits
│   ├── trajectory.py             # Event logging
│   ├── trajectory_analysis.py    # Strategy extraction
│   ├── strategy_cache.py         # Learn from success
│   ├── tool_bridge.py            # Controlled tool access
│   └── ...
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── property/                 # Hypothesis property tests
│   └── security/                 # Security tests
├── scripts/                      # Hook scripts
├── hooks/                        # hooks.json
├── commands/                     # Slash command definitions
└── docs/                         # Documentation
```

### Running Tests

The test suite has 3,200+ tests including property-based tests (Hypothesis) and benchmarks.

**Quick iteration** (~30 seconds):
```bash
# Fast mode - 10 examples per property test
HYPOTHESIS_PROFILE=fast uv run pytest tests/unit/ -v
```

**Standard development** (default, ~2-3 minutes):
```bash
# Dev mode - 50 examples per property test
uv run pytest tests/ -v
```

**Full CI run** (~5+ minutes):
```bash
# CI mode - 100 examples, all phases
HYPOTHESIS_PROFILE=ci uv run pytest tests/ -v
```

**Benchmarks** (run separately, ~1-2 minutes):
```bash
# Performance benchmarks
uv run pytest tests/benchmarks/ --benchmark-only
```

### Hypothesis Profiles

| Profile | Examples | Use Case |
|---------|----------|----------|
| `fast` | 10 | Quick iteration, TDD |
| `dev` | 50 | Local development (default) |
| `ci` | 100 | CI/CD, thorough testing |

Set via environment: `HYPOTHESIS_PROFILE=fast pytest ...`

### Test Categories

```bash
# Unit tests only (fastest)
uv run pytest tests/unit/ -v

# Integration tests
uv run pytest tests/integration/ -v

# Property-based tests
uv run pytest tests/property/ -v

# Security tests
uv run pytest tests/security/ -v

# Include benchmarks (excluded by default)
uv run pytest tests/ tests/benchmarks/ --benchmark-only
```

---

## Troubleshooting

### RLM Not Initializing

1. Check plugin installation: `claude plugin list`
2. Check hooks: `ls hooks/hooks.json`
3. Test init script: `uv run python scripts/init_rlm.py`

### Module Import Errors

Install dependencies:
```bash
uv sync --all-extras
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
| [Getting Started](./docs/getting-started.md) | Installation and first steps |
| [User Guide](./docs/user-guide.md) | Complete usage documentation |
| [Specification](./rlm-claude-code-spec.md) | Technical specification |
| [Architecture](./docs/process/architecture.md) | ADRs and design decisions |
| [SPEC Overview](./docs/spec/00-overview.md) | Capability specifications |

---

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601v1) - Zhang, Kraska, Khattab
- [Alex Zhang's RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/)
- [Claude Code Plugins](https://docs.anthropic.com/en/docs/claude-code)

---

## License

MIT
