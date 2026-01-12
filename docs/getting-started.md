# Getting Started with RLM-Claude-Code

This guide walks you through installing and using RLM-Claude-Code.

## Prerequisites

- **Python 3.12+**
- **uv** package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **Claude Code** (optional, for plugin usage)

Verify your setup:

```bash
uv --version
python --version
```

---

## Installation

### Clone and Install

```bash
git clone https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code

# Install all dependencies
uv sync --all-extras
```

### Verify Installation

```bash
# Run tests (should see 1000+ pass)
uv run pytest tests/ -v --tb=short

# Type check
uv run ty check src/
```

---

## As a Claude Code Plugin

If you want to use RLM with Claude Code:

```bash
# From the rlm-claude-code directory
claude plugins install . --scope user
```

Start Claude Code and you should see:
```
RLM initialized
```

Test it works:
```
/rlm status
```

---

## First Steps

### 1. Understand the REPL

RLM provides a sandboxed Python environment with context variables and helper functions:

```python
# Context is automatically available:
conversation  # List of messages
files         # Dict of filename → content
tool_outputs  # List of tool results
working_memory  # Scratchpad dict

# Helper functions:
peek(files['main.py'], 0, 500)  # View first 500 chars
search(files, 'def authenticate')  # Find patterns
llm("Summarize this code", context=files['auth.py'])  # Sub-query
```

### 2. Try the Slash Commands

| Command | What it does |
|---------|--------------|
| `/rlm` | Show current status |
| `/rlm on` | Enable RLM mode |
| `/rlm off` | Disable RLM mode |
| `/rlm mode fast` | Quick mode (depth=1, Haiku) |
| `/rlm mode thorough` | Deep mode (depth=3, Opus) |
| `/simple` | Bypass RLM for one query |

### 3. Understand Auto-Activation

RLM automatically activates for complex tasks:
- Large context (>80K tokens)
- Cross-file questions
- Debugging requests
- Architecture discussions

For simple questions, RLM stays off to avoid overhead.

---

## Configuration

RLM stores settings at `~/.claude/rlm-config.json`:

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

Edit directly or use slash commands:
```
/rlm mode thorough
/rlm depth 3
/rlm save
```

---

## Core Concepts

### Execution Modes

| Mode | Depth | Model | Best For |
|------|-------|-------|----------|
| `fast` | 1 | Haiku | Quick questions |
| `balanced` | 2 | Sonnet | Most tasks (default) |
| `thorough` | 3 | Opus | Complex debugging |

### Memory System

RLM can persist knowledge across sessions:

```python
from src import MemoryStore, MemoryEvolution

store = MemoryStore(db_path="~/.claude/rlm-memory.db")

# Store facts
store.create_node(
    node_type="fact",
    content="Project uses PostgreSQL",
    confidence=0.9,
)

# Memory evolves: task → session → longterm
evolution = MemoryEvolution(store)
evolution.consolidate(task_id="task-1")
evolution.promote(session_id="session-1")
```

### Budget Tracking

Control costs with limits:

```python
from src import EnhancedBudgetTracker, BudgetLimits

tracker = EnhancedBudgetTracker()
tracker.set_limits(BudgetLimits(
    max_cost_per_task=5.0,
    max_recursive_calls=10,
))
```

Or via slash command:
```
/rlm budget $5
```

---

## Troubleshooting

### Tests Failing

```bash
# Check dependencies
uv sync --all-extras

# Run with verbose output
uv run pytest tests/ -v --tb=long
```

### RLM Not Initializing (Plugin)

1. Verify installation: `claude plugins list`
2. Check hooks exist: `ls hooks/hooks.json`
3. Test manually: `uv run python scripts/init_rlm.py`

### Reset Everything

```bash
rm ~/.claude/rlm-config.json
rm ~/.claude/rlm-memory.db
```

---

## Next Steps

- [User Guide](./user-guide.md) - Complete usage documentation
- [Architecture](./process/architecture.md) - Design decisions
- [SPEC Overview](./spec/00-overview.md) - Capability specifications
- [README](../README.md) - Quick reference
