# Getting Started with RLM-Claude-Code

Choose your installation path based on how you plan to use RLM.

---

## Prerequisites (All Users)

- **Python 3.12+** — `brew install python@3.12` or [python.org](https://python.org)
- **uv** package manager — [install uv](https://docs.astral.sh/uv/getting-started/installation/)
- **Node.js 20+** — `brew install node` or [nodejs.org](https://nodejs.org)

```bash
# Verify
uv --version && python --version && node --version
```

---

## Installation Paths

### Path 1: Light User (Marketplace Install)

**Best for:** Users who want to just use RLM without building anything.

```bash
# Add marketplace and install
claude plugin marketplace add github:rand/rlm-claude-code
claude plugin install rlm-claude-code@rlm-claude-code
```

**What happens automatically:**
- On first session, the plugin checks dependencies
- Downloads pre-built binaries and wheel from GitHub releases
- Creates Python venv
- AI sees setup status and can help if issues found

**If something is missing**, the AI assistant will see the status and guide you.

---

### Path 2: Clone + Build

**Best for:** Users who want to clone the repo and set up manually.

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code

# One command setup (downloads pre-built binaries)
npm install

# Verify
npm run verify
```

**Then install as plugin:**
```bash
claude plugin install . --scope user
```

---

### Path 3: Developer (Local Development)

**Best for:** Developers working on RLM itself, using symlink to source.

```bash
# Clone and build everything from source
git clone --recurse-submodules https://github.com/your-fork/rlm-claude-code.git
cd rlm-claude-code

# Build from source (requires Rust + Go)
npm run build -- --all

# Create symlink for development
claude plugin install . --scope user
```

**What you get:**
- `bin/` contains locally built Go binaries
- `vendor/loop/rlm-core/target/wheels/` contains locally built wheel
- venv uses the local wheel (no pip install needed)
- Changes to source are immediately reflected

---

## Verification

All users can verify their setup:

```bash
# Check setup status
npm run ensure-setup

# Verify everything works
npm run verify

# Run tests
npm run test
```

---

## Self-Healing System

The plugin includes automatic dependency checking:

| Check | What's Verified |
|-------|-----------------|
| `uv` | Package manager installed |
| `venv` | Python virtual environment exists |
| `binaries` | Go hook binaries for your platform |
| `rlm_core` | Python package installed |

**How it works:**
1. On SessionStart, `ensure-setup.ts` runs
2. Outputs JSON status to AI via hooks
3. If `needsAttention: true`, AI guides user to fix

**Manual commands:**
```bash
npm run ensure-setup           # Check status
npm run ensure-setup -- --fix  # Auto-fix (downloads from GitHub)
```

---

## First Steps

### 1. Try the REPL

RLM provides a sandboxed Python environment:

```python
# Context is automatically available:
conversation  # List of messages
files         # Dict of filename → content
tool_outputs  # List of tool results

# Helper functions:
peek(files['main.py'], 0, 500)     # View slice
search(files, 'def authenticate')  # Find patterns
llm("Summarize this", context=file)  # Sub-query
```

### 2. Try Slash Commands

```bash
/rlm-claude-code:rlm              # Show status
/rlm-claude-code:rlm activate     # Launch RLM now
/rlm-claude-code:rlm-orchestrator # Complex context tasks
```

### 3. Understand Auto-Activation

RLM automatically activates for:
- Large context (>80K tokens)
- Cross-file questions
- Debugging requests
- Architecture discussions

---

## Configuration

Settings at `~/.claude/rlm-config.json`:

```json
{
  "activation": { "mode": "complexity" },
  "depth": { "default": 2, "max": 3 },
  "trajectory": { "verbosity": "normal" }
}
```

---

## Troubleshooting

### Setup Issues

```bash
# Check what's missing
npm run ensure-setup

# Auto-fix (downloads from GitHub)
npm run ensure-setup -- --fix
```

### Tests Failing

```bash
uv sync --all-extras
uv run pytest tests/ -v --tb=long
```

### Reset Everything

```bash
rm ~/.claude/rlm-config.json
rm ~/.claude/rlm-memory.db
```

---

## Next Steps

- [User Guide](./user-guide.md) — Complete usage documentation
- [Architecture](./process/architecture.md) — Design decisions
- [SPEC Overview](./spec/00-overview.md) — Capability specifications
