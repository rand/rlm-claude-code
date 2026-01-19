# Session Isolation for Multi-Terminal Workflows

## Problem

When running multiple Claude Code instances on the same project (e.g., one terminal working on auth, another on billing), memories from different tasks can cross-contaminate. Terminal B might retrieve memories stored by Terminal A, leading to confusion and incorrect context.

## Solution

Session isolation filters `task` and `session` tier memories by `session_id`, ensuring each terminal only sees its own memories. Shared knowledge (`longterm` and `archive` tiers) remains accessible to all sessions.

### Tier Behavior

| Tier | Isolation | Use Case |
|------|-----------|----------|
| `task` | Per-session | Current task context, ephemeral |
| `session` | Per-session | Session-specific learning |
| `longterm` | Shared | Cross-session knowledge (e.g., "DB runs on port 5432") |
| `archive` | Shared | Historical reference |

## API Changes

### `MemoryStore.query_nodes()`

New optional parameter:

```python
def query_nodes(
    self,
    node_type: str | None = None,
    tier: str | None = None,
    min_confidence: float = 0.0,
    limit: int = 100,
    session_id: str | None = None,  # NEW
    include_archived: bool = False,
) -> list[Node]:
```

When `session_id` is provided and tier is `task` or `session`, results are filtered to nodes where `metadata.session_id` matches.

### `ReplEnvironment` Methods

All memory methods now accept optional `session_id`:

```python
# Storing with session context
memory_add_fact(content, tier="session", session_id="abc123")
memory_add_experience(content, outcome, session_id="abc123")

# Querying with session isolation
memory_query(query, tier="session", session_id="abc123")
memory_get_context(session_id="abc123")
```

## Usage

### Storing Session-Scoped Memories

```python
from rlm import MemoryStore

store = MemoryStore()
session_id = "session-abc123"  # From Claude Code session

# Store a task-specific fact
store.add_fact(
    content="Working on JWT authentication",
    tier="task",
    metadata={"session_id": session_id}
)
```

### Querying with Isolation

```python
# Only returns nodes from this session
results = store.query_nodes(
    tier="session",
    session_id=session_id,
    limit=10
)
```

### Storing Shared Knowledge

```python
# Longterm facts are visible to all sessions
store.add_fact(
    content="PostgreSQL runs on port 5432",
    tier="longterm",
    metadata={"session_id": session_id}  # Origin tracked, but not filtered
)
```

## Implementation Details

Session filtering uses SQLite's `json_extract()` on the existing `metadata` JSON column:

```sql
SELECT * FROM nodes
WHERE tier = 'session'
  AND json_extract(metadata, '$.session_id') = ?
ORDER BY confidence DESC, last_accessed DESC
LIMIT ?
```

This approach requires no schema migration and maintains backward compatibility.

## Backward Compatibility

- `session_id` parameter defaults to `None`
- When `None`, all nodes are returned (existing behavior)
- Nodes without `session_id` in metadata remain visible to all sessions
- No changes required for single-terminal workflows

## Testing

Run session isolation tests:

```bash
uv run pytest tests/unit/test_session_isolation.py -v
uv run pytest tests/integration/test_parallel_sessions.py -v
```
