# RLM-Claude-Code Documentation

## User Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](../getting-started.md) | Installation, setup, and first steps |
| [User Guide](../user-guide.md) | Complete usage documentation |
| [README](../../README.md) | Quick start and overview |

## Technical Documentation

| Document | Description |
|----------|-------------|
| [Specification](../../rlm-claude-code-spec.md) | Full technical specification |
| [Architecture Decisions](./architecture.md) | ADRs and design rationale |
| [CLAUDE.md](../../CLAUDE.md) | Claude Code session guide |

## Development Documentation

| Document | Description |
|----------|-------------|
| [Implementation](./implementation.md) | Implementation workflow and phases |
| [Full-System Empirical Validation](./full-system-empirical-validation.md) | End-to-end validation matrix, evidence, and remaining gaps |
| [Code Review](./code-review.md) | Code review checklist and standards |
| [Testing](./testing.md) | Testing strategy and requirements |
| [Debugging](./debugging.md) | Debugging workflow for RLM issues |

---

## Quick Reference

### For New Users

1. Start with the [Getting Started Guide](../getting-started.md)
2. Read the [User Guide](../user-guide.md) for detailed usage
3. Check the [README](../../README.md) for quick reference

### For Contributors

1. Read the [Specification](../../rlm-claude-code-spec.md) first
2. Check [Implementation](./implementation.md) for current phase
3. Follow [Code Review](./code-review.md) checklist before PRs
4. Use [Architecture](./architecture.md) for design decisions

### Before Starting Work

```bash
# Ensure dependencies are current
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Type check
uv run ty check src/
```

### During Implementation

```bash
# Type check continuously
uv run ty check src/

# Lint before committing
uv run ruff check src/ --fix
uv run ruff format src/

# Run tests
uv run pytest tests/ -v
```

### Before Committing

1. Run `/code-review` command
2. Ensure all tests pass
3. Update trajectory examples if behavior changed
4. Document any new patterns in architecture.md

---

## Core Principles

1. **Spec-Driven**: The spec is the source of truth. Implementation diverging from spec requires spec update first.

2. **Incremental Verification**: Each component should be testable in isolation before integration.

3. **Trajectory-First**: When debugging, always capture and analyze the trajectory before making changes.

4. **Type Everything**: All public APIs must have full type annotations validated by `ty`.

---

## Key Paths

| Path | Purpose |
|------|---------|
| `src/` | Source code |
| `tests/` | Test suite |
| `docs/` | Documentation |
| `scripts/` | Utility scripts |
| `hooks/` | Claude Code hooks |
| `.claude/commands/` | Slash commands |
