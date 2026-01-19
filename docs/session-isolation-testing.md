# Session Isolation Testing Guide

This guide walks through testing session isolation with two Claude Code terminals.

## Prerequisites

1. Hooks configured at `~/.claude/hooks.json` pointing to local fork
2. Virtual environment set up at `~/src/rlm-claude-code/.venv`

## Setup

Open two terminal windows. We'll call them **Terminal A** and **Terminal B**.

---

## Terminal A: Authentication Feature

### Step 1: Start Claude Code
```bash
cd ~/src/narai
claude
```

### Step 2: Store task-tier facts (isolated to this session)
Ask Claude to run:
```python
# In the REPL or via a tool
from src.memory_store import MemoryStore
store = MemoryStore()

# Add task-tier fact (should be isolated)
store.create_node(
    node_type="fact",
    content="Working on JWT authentication",
    tier="task",
    metadata={"session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown")}
)
```

Or simpler - just tell Claude:
> "Remember that I'm working on JWT authentication for this task"

### Step 3: Store session-tier facts
> "Remember for this session: Auth service uses port 3001"

### Step 4: Store longterm fact (shared across sessions)
> "Add to longterm memory: PostgreSQL database runs on port 5433"

---

## Terminal B: Billing Feature

### Step 1: Start Claude Code (new session)
```bash
cd ~/src/narai
claude
```

### Step 2: Store task-tier facts
> "Remember that I'm working on Stripe billing integration"

### Step 3: Store session-tier facts
> "Remember for this session: Billing service uses port 3002"

---

## Verification Tests

### Test 1: Task Tier Isolation (in Terminal A)
Ask Claude:
> "What am I currently working on?"

**Expected:** Should mention JWT authentication
**Should NOT mention:** Stripe billing

### Test 2: Task Tier Isolation (in Terminal B)
Ask Claude:
> "What am I currently working on?"

**Expected:** Should mention Stripe billing
**Should NOT mention:** JWT authentication

### Test 3: Session Tier Isolation (in Terminal A)
Ask Claude:
> "What port does my service use?"

**Expected:** Port 3001 (auth)
**Should NOT mention:** Port 3002 (billing)

### Test 4: Longterm Tier Sharing (in Terminal B)
Ask Claude:
> "What port does PostgreSQL run on?"

**Expected:** Port 5433 (should see the fact stored from Terminal A)

---

## Expected Results Summary

| Test | Terminal | Query | Expected Result |
|------|----------|-------|-----------------|
| Task isolation | A | "What am I working on?" | JWT auth only |
| Task isolation | B | "What am I working on?" | Stripe billing only |
| Session isolation | A | "What port?" | 3001 only |
| Session isolation | B | "What port?" | 3002 only |
| Longterm sharing | B | "PostgreSQL port?" | 5433 (from A) |

---

## Debugging

### Check session ID
In each terminal, ask Claude to run:
```bash
echo $CLAUDE_SESSION_ID
```

Each terminal should have a different session ID.

### Check stored memories
```python
from src.memory_store import MemoryStore
store = MemoryStore()

# See all nodes
nodes = store.query_nodes(limit=20)
for n in nodes:
    print(f"[{n.tier}] {n.content[:50]}... session={n.metadata.get('session_id', 'none')}")
```

### Check isolation is working
```python
# Query with session filter
session_id = os.environ.get("CLAUDE_SESSION_ID")
task_nodes = store.query_nodes(tier="task", session_id=session_id)
print(f"Task nodes for this session: {len(task_nodes)}")
```
