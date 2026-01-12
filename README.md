# RLM-Claude-Code

Transform Claude Code into a Recursive Language Model (RLM) agent with intelligent orchestration, unbounded context handling, persistent memory, and REPL-based decomposition.

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

```bash
# Clone the repository
git clone https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code

# Install dependencies
uv sync --all-extras

# Run tests to verify setup
uv run pytest tests/ -v
```

### As a Claude Code Plugin

```bash
# Install as a local plugin
claude plugins install . --scope user
```

After installation, start Claude Code and you should see "RLM initialized" on startup.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              INTELLIGENT ORCHESTRATOR                    │
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
│                 RLM EXECUTION ENGINE                     │
│                                                          │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │  Context Manager │    │     REPL Sandbox         │   │
│  │  • Externalize   │───►│  • peek(), search()      │   │
│  │    conversation  │    │  • llm(), llm_batch()    │   │
│  │  • files, tools  │    │  • map_reduce()          │   │
│  └──────────────────┘    │  • find_relevant()       │   │
│                          │  • memory_*() functions  │   │
│                          └──────────────────────────┘   │
│                                     │                    │
│                                     ▼                    │
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
│                  PERSISTENCE LAYER                       │
│                                                          │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │  Memory Store    │    │   Reasoning Traces       │   │
│  │  • Facts, exps   │    │  • Goals, decisions      │   │
│  │  • Hyperedges    │    │  • Options, outcomes     │   │
│  │  • SQLite + WAL  │    │  • Decision trees        │   │
│  └──────────────────┘    └──────────────────────────┘   │
│           │                         │                    │
│           ▼                         ▼                    │
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
│                  BUDGET & TRAJECTORY                     │
│  • Token tracking per component                          │
│  • Cost limits with alerts                               │
│  • Streaming trajectory output                           │
│  • JSON export for analysis                              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Final Answer
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
| `run_tool(cmd, args)` | Safe subprocess (ty, ruff only) |

**Memory Functions** (when enabled):

| Function | Description |
|----------|-------------|
| `memory_query(query, limit)` | Search stored knowledge |
| `memory_add_fact(content, confidence)` | Store a fact |
| `memory_add_experience(content, outcome, success)` | Store an experience |
| `memory_get_context(limit)` | Retrieve relevant context |
| `memory_relate(node1, node2, relation)` | Create relationships |

### Memory System

Persistent storage for cross-session learning:

- **Node Types**: facts, experiences, procedures, goals
- **Memory Tiers**: task → session → longterm → archive
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

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API access (uses Claude Code's key) |
| `OPENAI_API_KEY` | OpenAI API access (optional, for GPT models) |
| `RLM_CONFIG_PATH` | Custom config file location |
| `RLM_DEBUG` | Enable debug logging |

---

## Hooks

RLM integrates with Claude Code via hooks:

| Hook | Script | Purpose |
|------|--------|---------|
| `SessionStart` | `init_rlm.py` | Initialize RLM environment |
| `UserPromptSubmit` | `check_complexity.py` | Decide if RLM should activate |
| `PreToolUse` | `sync_context.py` | Sync tool context with RLM state |
| `PostToolUse` | `capture_output.py` | Capture tool output for context |
| `PreCompact` | `externalize_context.py` | Externalize before compaction |
| `Stop` | `save_trajectory.py` | Save trajectory on session end |

---

## Development

### Setup

```bash
git clone https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code

# Install all dependencies
uv sync --all-extras

# Run tests (1000+ tests)
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

### Test Categories

```bash
# Unit tests
uv run pytest tests/unit/ -v

# Integration tests
uv run pytest tests/integration/ -v

# Property-based tests
uv run pytest tests/property/ -v -m hypothesis

# Security tests
uv run pytest tests/security/ -v

# Benchmarks
uv run pytest tests/benchmarks/ --benchmark-only
```

---

## Troubleshooting

### RLM Not Initializing

1. Check plugin installation: `claude plugins list`
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
