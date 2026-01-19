# Session Isolation Testing Guide

This guide walks through testing session isolation with two Claude Code terminals.

## Overview

Session isolation ensures that **task** and **session** tier memories are scoped to individual Claude Code sessions. This prevents cross-contamination when running multiple terminals on the same project.

**What's isolated:**
- `task` tier: Ephemeral facts for the current task
- `session` tier: Facts that persist for the session duration

**Note:** This implementation focuses on task/session isolation. Longterm memory features are not currently active.

---

## Prerequisites

### 1. Clone the fork with session isolation

```bash
git clone https://github.com/narailabs/rlm-claude-code.git ~/src/rlm-claude-code
cd ~/src/rlm-claude-code
git checkout feature/session-isolation
```

### 2. Install dependencies

```bash
cd ~/src/rlm-claude-code
uv sync --all-extras --python 3.12
```

Verify the installation:
```bash
.venv/bin/python -c "from src.memory_store import MemoryStore; print('OK')"
```

### 3. Configure Claude Code hooks

Create or update `~/.claude/hooks.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "cd /Users/YOUR_USERNAME/src/rlm-claude-code && .venv/bin/python scripts/init_rlm.py",
            "timeout": 5000,
            "description": "Initialize RLM environment"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "cd /Users/YOUR_USERNAME/src/rlm-claude-code && .venv/bin/python scripts/sync_context.py",
            "timeout": 2000,
            "description": "Sync tool context with RLM state"
          }
        ]
      }
    ]
  }
}
```

**Important:** Replace `YOUR_USERNAME` with your actual username.

### 4. Verify hooks are loaded

Start a new Claude Code session and check for the "RLM initialized" message, or run:
```bash
cat ~/.claude/rlm-config.json
```

---

## Quick Test (Programmatic)

Run this script to verify session isolation works:

```bash
cd ~/src/rlm-claude-code && .venv/bin/python << 'EOF'
import tempfile, os
from src.memory_store import MemoryStore

with tempfile.TemporaryDirectory() as tmpdir:
    store = MemoryStore(db_path=os.path.join(tmpdir, "test.db"))

    # Terminal A adds a task fact
    store.add_fact("Working on JWT auth", tier="task",
                   metadata={"session_id": "terminal-a"})

    # Terminal B adds a task fact
    store.add_fact("Working on Stripe billing", tier="task",
                   metadata={"session_id": "terminal-b"})

    # Query from each perspective
    a_sees = store.query_nodes(tier="task", session_id="terminal-a")
    b_sees = store.query_nodes(tier="task", session_id="terminal-b")

    print(f"Terminal A sees: {[n.content for n in a_sees]}")
    print(f"Terminal B sees: {[n.content for n in b_sees]}")

    assert len(a_sees) == 1 and "JWT" in a_sees[0].content
    assert len(b_sees) == 1 and "Stripe" in b_sees[0].content
    print("✓ Session isolation working!")
EOF
```

---

## Manual Two-Terminal Test

### Setup

Open two terminal windows: **Terminal A** and **Terminal B**.

### Terminal A: Authentication Feature

**Step 1:** Start Claude Code
```bash
cd ~/src/narai
claude
```

**Step 2:** Store task-tier fact
> "Remember that I'm working on JWT authentication for this task"

**Step 3:** Store session-tier fact
> "Remember for this session: Auth service uses port 3001"

### Terminal B: Billing Feature

**Step 1:** Start Claude Code (new session)
```bash
cd ~/src/narai
claude
```

**Step 2:** Store task-tier fact
> "Remember that I'm working on Stripe billing integration"

**Step 3:** Store session-tier fact
> "Remember for this session: Billing service uses port 3002"

---

## Verification Tests

### Test 1: Task Tier Isolation (Terminal A)
Ask Claude:
> "What am I currently working on?"

**Expected:** JWT authentication
**Should NOT see:** Stripe billing

### Test 2: Task Tier Isolation (Terminal B)
Ask Claude:
> "What am I currently working on?"

**Expected:** Stripe billing
**Should NOT see:** JWT authentication

### Test 3: Session Tier Isolation (Terminal A)
Ask Claude:
> "What port does my service use?"

**Expected:** Port 3001
**Should NOT see:** Port 3002

### Test 4: Session Tier Isolation (Terminal B)
Ask Claude:
> "What port does my service use?"

**Expected:** Port 3002
**Should NOT see:** Port 3001

---

## Expected Results Summary

| Test | Terminal | Query | Expected Result |
|------|----------|-------|-----------------|
| Task isolation | A | "What am I working on?" | JWT auth only |
| Task isolation | B | "What am I working on?" | Stripe billing only |
| Session isolation | A | "What port?" | 3001 only |
| Session isolation | B | "What port?" | 3002 only |

---

## Debugging

### Check session ID
```bash
echo $CLAUDE_SESSION_ID
```
Each terminal should have a different session ID.

### Check stored memories
```python
from src.memory_store import MemoryStore
import os

store = MemoryStore()
session_id = os.environ.get("CLAUDE_SESSION_ID")

# See this session's task nodes
task_nodes = store.query_nodes(tier="task", session_id=session_id)
print(f"Task nodes for this session: {len(task_nodes)}")
for n in task_nodes:
    print(f"  - {n.content}")

# See this session's session nodes
session_nodes = store.query_nodes(tier="session", session_id=session_id)
print(f"Session nodes: {len(session_nodes)}")
for n in session_nodes:
    print(f"  - {n.content}")
```

### Verify no cross-contamination
```python
# Query without session filter (admin view)
all_nodes = store.query_nodes(tier="task")
print(f"Total task nodes across all sessions: {len(all_nodes)}")
```
