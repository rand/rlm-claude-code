# RLM-Claude-Code

Transform Claude Code into a Recursive Language Model (RLM) agent with intelligent multi-provider routing, unbounded context handling, and REPL-based decomposition.

## Installation

### As a Claude Code Plugin (Recommended)

```bash
# Clone the repository
git clone https://github.com/heroku/rlm-claude-code.git
cd rlm-claude-code

# Install as a plugin
claude plugin install . --scope user

# Set up API keys
./scripts/set-api-key.sh
```

### Manual Installation

```bash
# Clone and install dependencies
git clone https://github.com/heroku/rlm-claude-code.git
cd rlm-claude-code
uv sync

# Configure API keys
./scripts/set-api-key.sh anthropic YOUR_ANTHROPIC_KEY
./scripts/set-api-key.sh openai YOUR_OPENAI_KEY

# Verify installation
uv run pytest tests/ -v
```

## Features

### Smart Model Routing

Automatically routes queries to the optimal model based on task type:

| Query Type | Primary Model | Provider |
|------------|---------------|----------|
| Code, Debugging | GPT-5.2-Codex | OpenAI |
| Planning, Architecture | Opus | Anthropic |
| Analytical | Opus | Anthropic |
| Factual, Simple | Haiku | Anthropic |

### RLM Context Management

- **Complexity-based activation**: RLM activates when tasks require cross-file reasoning
- **Depth=2 recursion**: Root → Analysis → Verification pattern
- **Context externalization**: Large contexts become Python variables in a REPL
- **Streaming trajectory**: See reasoning unfold in real-time

## Slash Commands

After installation, these commands are available in Claude Code:

| Command | Description |
|---------|-------------|
| `/rlm` | Toggle or configure RLM mode |
| `/trajectory` | Analyze an RLM trajectory file |
| `/simple` | Bypass RLM for simple operations |
| `/test` | Run the test suite |
| `/bench` | Run performance benchmarks |
| `/code-review` | Review code changes |

## Configuration

Configuration is stored at `~/.claude/rlm-config.json`:

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
  },
  "models": {
    "root_model": "opus",
    "recursive_depth_1": "sonnet",
    "recursive_depth_2": "haiku",
    "openai_root": "gpt-5.2-codex",
    "openai_recursive": "gpt-4o-mini"
  }
}
```

## Architecture

```
User Query
    ↓
Smart Router → Select optimal model (Opus/Codex/Haiku)
    ↓
Complexity Classifier
    ↓ (if complex)
RLM Orchestrator
    ├── Context Manager → Externalize to Python vars
    ├── REPL Environment → Execute peek/search/summarize
    └── Recursive Handler → Spawn sub-queries (depth≤2)
    ↓
Trajectory Stream → User sees reasoning
    ↓
Claude Code Tools → bash/edit/read as normal
    ↓
Final Answer
```

## Development

```bash
# Type checking
uv run ty check src/

# Linting
uv run ruff check src/ --fix
uv run ruff format src/

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src/

# Run benchmarks
uv run pytest tests/benchmarks/ --benchmark-only
```

## Documentation

| Document | Purpose |
|----------|---------|
| [rlm-claude-code-spec.md](./rlm-claude-code-spec.md) | Full specification |
| [CLAUDE.md](./CLAUDE.md) | Claude Code session guide |
| [docs/process/](./docs/process/) | Development process |

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601v1) - Zhang, Kraska, Khattab
- [Claude Code Plugins](https://code.claude.com/docs/en/plugins)
- [CPMpy](https://cpmpy.readthedocs.io/)

## License

MIT
