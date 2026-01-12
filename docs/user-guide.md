# RLM-Claude-Code User Guide

Complete documentation for using RLM-Claude-Code effectively.

## Table of Contents

- [Understanding RLM](#understanding-rlm)
- [REPL Environment](#repl-environment)
- [Slash Commands](#slash-commands)
- [Execution Modes](#execution-modes)
- [Auto-Activation](#auto-activation)
- [Memory System](#memory-system)
- [Reasoning Traces](#reasoning-traces)
- [Budget Management](#budget-management)
- [Trajectory Analysis](#trajectory-analysis)
- [Strategy Learning](#strategy-learning)
- [Advanced Configuration](#advanced-configuration)
- [Best Practices](#best-practices)

---

## Understanding RLM

### The Problem

Large Language Models have context limits. Even with 200K token windows, Claude can struggle with:

- **Information overload**: Too much context dilutes attention
- **Cross-reference reasoning**: Connecting information across distant parts
- **Systematic analysis**: Ensuring nothing is missed in large codebases

### The Solution

RLM (Recursive Language Model) solves this by decomposition:

1. **Context Externalization**: Large contexts become Python variables
2. **REPL Environment**: Claude writes code to explore context programmatically
3. **Recursive Sub-Queries**: Complex questions spawn focused sub-queries
4. **Memory Persistence**: Facts and experiences persist across sessions
5. **Strategy Learning**: Successful patterns are remembered for similar tasks

### Example Flow

```
User: "Find security vulnerabilities in the auth module"

RLM Analysis:
├─ Complexity classifier detects cross-file reasoning needed
├─ Orchestrator chooses: depth=2, model=sonnet, tools=read_only
├─ Context externalized: auth/*.py files as Python dict
├─ REPL execution:
│   ├─ peek(files['auth/handler.py'][:500])
│   ├─ search(files, 'password', regex=False)
│   └─ find_relevant(files['auth/session.py'], 'validation')
├─ Sub-queries spawned:
│   ├─ llm("Analyze input validation", files['handler.py'])
│   └─ llm("Check session management", files['session.py'])
├─ Results aggregated
└─ Final response with findings
```

---

## REPL Environment

The REPL is a sandboxed Python environment for context manipulation.

### Context Variables

| Variable | Type | Description |
|----------|------|-------------|
| `conversation` | `list[dict]` | Messages with `role` and `content` |
| `files` | `dict[str, str]` | Filename → content mapping |
| `tool_outputs` | `list[dict]` | Tool results with `tool` and `content` |
| `working_memory` | `dict` | Scratchpad for intermediate results |

### Helper Functions

#### `peek(var, start=0, end=1000)`

View a slice of any context variable.

```python
# First 500 chars of a file
peek(files['main.py'], 0, 500)

# Middle of conversation
peek(conversation, 5, 10)

# First 3 items of a dict
peek(files, 0, 3)
```

#### `search(var, pattern, regex=False)`

Find patterns in context. Returns list of matches with location info.

```python
# Find all authentication-related code
search(files, 'authenticate')

# Regex search for function definitions
search(files['utils.py'], r'def \w+\(', regex=True)

# Search conversation for error mentions
search(conversation, 'error')
```

#### `summarize(var, max_tokens=500)`

LLM-powered summarization via sub-call.

```python
# Summarize a large file
summary = summarize(files['large_module.py'], max_tokens=200)
```

#### `llm(query, context=None, spawn_repl=False)`

Spawn a recursive sub-query.

```python
# Simple sub-query
result = llm("What does this function do?", files['auth.py'])

# With REPL access for the sub-query
result = llm("Analyze this module", files['complex.py'], spawn_repl=True)
```

#### `llm_batch(queries, spawn_repl=False)`

Execute multiple LLM queries in parallel.

```python
# Analyze multiple modules concurrently
results = llm_batch([
    ("Analyze auth module", files['auth.py']),
    ("Analyze db module", files['db.py']),
    ("Analyze api module", files['api.py']),
])
```

#### `map_reduce(content, map_prompt, reduce_prompt, n_chunks=4, model="auto")`

Apply map-reduce pattern to large content.

```python
# Analyze large file by chunks
result = map_reduce(
    large_file_content,
    map_prompt="Find potential bugs in this code chunk",
    reduce_prompt="Combine these findings into a prioritized list",
    n_chunks=4,
    model="fast",
)
```

#### `find_relevant(content, query, top_k=5, use_llm_scoring=False)`

Find sections most relevant to a query.

```python
# Find authentication-related sections
relevant = find_relevant(
    files['large_module.py'],
    query="password validation",
    top_k=3,
)
# Returns: [(chunk, score), ...]
```

#### `extract_functions(content)`

Parse and extract function definitions.

```python
# Get all functions from a file
functions = extract_functions(files['utils.py'])
# Returns: [{'name': 'foo', 'args': [...], 'body': '...', 'line': 42}, ...]
```

#### `run_tool(cmd, args=[])`

Execute safe subprocess commands (limited to `ty`, `ruff`).

```python
# Type check a file
result = run_tool("ty", ["check", "src/module.py"])

# Lint a file
result = run_tool("ruff", ["check", "src/module.py"])
```

### Memory Functions

When memory is enabled, additional functions are available:

#### `memory_query(query, limit=10)`

Search stored knowledge.

```python
# Find facts about authentication
results = memory_query("authentication patterns", limit=5)
```

#### `memory_add_fact(content, confidence=0.5)`

Store a fact.

```python
# Remember a discovery
memory_add_fact("This project uses JWT for auth", confidence=0.9)
```

#### `memory_add_experience(content, outcome, success)`

Store an experience with outcome.

```python
# Record what worked
memory_add_experience(
    "Used map_reduce for large file analysis",
    "Successfully identified 3 bugs",
    success=True,
)
```

#### `memory_get_context(limit=10)`

Get recent/relevant context nodes.

```python
# Get context for current work
context_nodes = memory_get_context(limit=5)
```

#### `memory_relate(node1_id, node2_id, relation)`

Create relationships between nodes.

```python
# Link related facts
memory_relate(fact1_id, fact2_id, "supports")
```

---

## Slash Commands

### Core Commands

| Command | Description |
|---------|-------------|
| `/rlm` | Show current RLM status |
| `/rlm on` | Enable RLM for this session |
| `/rlm off` | Disable RLM mode |
| `/rlm status` | Show detailed configuration |

### Mode Commands

| Command | Description |
|---------|-------------|
| `/rlm mode fast` | Quick, shallow analysis |
| `/rlm mode balanced` | Standard processing (default) |
| `/rlm mode thorough` | Deep, comprehensive analysis |

### Configuration Commands

| Command | Description |
|---------|-------------|
| `/rlm depth <0-3>` | Set maximum recursion depth |
| `/rlm budget $X` | Set session cost limit |
| `/rlm model <name>` | Force model (opus/sonnet/haiku/auto) |
| `/rlm tools <level>` | Tool access (none/repl/read/full) |
| `/rlm verbosity <level>` | Output detail (minimal/normal/verbose/debug) |
| `/rlm reset` | Reset all settings to defaults |
| `/rlm save` | Save current preferences to disk |

### Other Commands

| Command | Description |
|---------|-------------|
| `/simple` | Bypass RLM for current query only |
| `/trajectory <file>` | Analyze a saved trajectory file |
| `/test` | Run the test suite |
| `/bench` | Run performance benchmarks |
| `/code-review` | Review code changes |

---

## Execution Modes

### Fast Mode

```
/rlm mode fast
```

| Setting | Value |
|---------|-------|
| Depth | 1 |
| Model | Haiku |
| Tools | REPL only |

**Best for**: Quick questions, iteration, simple tasks.

### Balanced Mode (Default)

```
/rlm mode balanced
```

| Setting | Value |
|---------|-------|
| Depth | 2 |
| Model | Sonnet |
| Tools | Read-only |

**Best for**: Most daily tasks, feature development, bug fixes.

### Thorough Mode

```
/rlm mode thorough
```

| Setting | Value |
|---------|-------|
| Depth | 3 |
| Model | Opus |
| Tools | Full access |

**Best for**: Security audits, architecture decisions, complex debugging.

---

## Auto-Activation

### How It Works

RLM analyzes each query to decide whether to activate:

1. **Context Size**: Large contexts (>80K tokens) trigger activation
2. **Query Complexity**: Cross-file references, debugging keywords
3. **Pattern Matching**: Architecture questions, comparison requests
4. **User Preference**: Manual `/rlm on` overrides everything

### Complexity Signals

| Signal | Examples |
|--------|----------|
| Cross-file reference | "How does auth.py interact with api.py?" |
| Debugging keywords | "Why does this fail?", "trace the error" |
| Architecture questions | "How should I structure this?" |
| Comparison requests | "What's the difference between X and Y?" |
| Multi-step tasks | "Refactor and add tests" |

### Controlling Activation

```
/rlm on          # Force activation for all queries
/rlm off         # Disable auto-activation
/simple          # Skip activation for one query
```

### Viewing Decisions

With debug verbosity:
```
/rlm verbosity debug
```

You'll see activation reasoning:
```
[ACTIVATION] Analyzing query...
  - Token count: 145,230 (above threshold)
  - Cross-file references: 3 detected
  - Complexity score: 0.87
  - Decision: ACTIVATE
```

---

## Memory System

RLM includes a persistent memory system for cross-session learning.

### Node Types

| Type | Description |
|------|-------------|
| `fact` | Verified information about the codebase |
| `experience` | Past actions and their outcomes |
| `procedure` | Known working approaches |
| `goal` | Tracked objectives |

### Memory Tiers

Memory evolves through tiers based on usage and confidence:

```
task → session → longterm → archive
```

| Tier | Lifespan | Purpose |
|------|----------|---------|
| `task` | Current task | Working memory |
| `session` | Current session | Short-term recall |
| `longterm` | Persistent | Core knowledge |
| `archive` | Compressed | Historical reference |

### Using Memory Programmatically

```python
from src import MemoryStore, MemoryEvolution

# Create store
store = MemoryStore(db_path="~/.claude/rlm-memory.db")

# Store facts
fact_id = store.create_node(
    node_type="fact",
    content="This project uses PostgreSQL 15",
    tier="task",
    confidence=0.9,
)

# Create relationships
store.create_edge(
    edge_type="relation",
    label="uses",
    members=[
        {"node_id": project_id, "role": "subject", "position": 0},
        {"node_id": fact_id, "role": "object", "position": 1},
    ],
)

# Evolve memory
evolution = MemoryEvolution(store)
evolution.consolidate(task_id="current-task")  # task → session
evolution.promote(session_id="session-1")  # session → longterm
evolution.decay(days_threshold=30)  # old → archive
```

---

## Reasoning Traces

Track decision-making for transparency and debugging.

### Creating Traces

```python
from src import ReasoningTraces

traces = ReasoningTraces(store)

# Create a goal
goal_id = traces.create_goal(
    content="Implement user authentication",
    prompt="How should I implement user authentication?",
    files=["src/auth.py", "src/models/user.py"],
)

# Create a decision point
decision_id = traces.create_decision(
    goal_id=goal_id,
    content="Choose authentication strategy",
)

# Add options
jwt_option = traces.add_option(decision_id, "Use JWT tokens")
session_option = traces.add_option(decision_id, "Use session cookies")

# Record the choice
traces.choose_option(decision_id, jwt_option)
traces.reject_option(decision_id, session_option, "JWT is more scalable for API")

# Create action and outcome
action_id = traces.create_action(decision_id, "Implementing JWT authentication")
outcome_id = traces.create_outcome(action_id, "JWT auth implemented successfully", success=True)
```

### Querying Traces

```python
# Get full decision tree
tree = traces.get_decision_tree(goal_id)

# Get rejected options with reasons
rejected = traces.get_rejected_options(decision_id)
for opt in rejected:
    print(f"Rejected: {opt.content} - {opt.reason}")
```

---

## Budget Management

### Setting Budgets

```
/rlm budget $5        # Session budget of $5
/rlm budget $0.50     # Budget of 50 cents
```

### How Budgets Work

- Budgets are per-session (reset when Claude Code restarts)
- RLM tracks estimated cost of each operation
- When budget is exceeded, RLM uses simpler strategies
- You're warned before exceeding budget

### Programmatic Budget Control

```python
from src import EnhancedBudgetTracker, BudgetLimits

tracker = EnhancedBudgetTracker()

# Set limits
tracker.set_limits(BudgetLimits(
    max_cost_per_task=5.0,
    max_recursive_calls=10,
    max_depth=3,
    max_repl_executions=50,
))

# Start tracking a task
tracker.start_task("analyze-codebase")
tracker.start_timing()

# Check before operations
allowed, reason = tracker.can_make_llm_call()
if not allowed:
    print(f"Blocked: {reason}")

# Record operations
tracker.record_llm_call(
    input_tokens=1000,
    output_tokens=500,
    model="sonnet",
    component=CostComponent.RECURSIVE_CALL,
)
tracker.record_repl_execution()
tracker.record_depth(2)

# Get metrics
metrics = tracker.get_metrics()
print(f"Cost: ${metrics.total_cost_usd:.2f}")
print(f"Calls: {metrics.sub_call_count}")
print(f"Max depth: {metrics.max_depth_reached}")

# End task
tracker.stop_timing()
tracker.end_task()
```

### Budget Alerts

The tracker can trigger alerts:

```python
tracker.set_limits(BudgetLimits(
    max_cost_per_task=5.0,
    warn_at_cost=4.0,  # Warn at 80%
))

# Check for alerts
alerts = tracker.get_alerts()
for alert in alerts:
    print(f"[{alert.level}] {alert.message}")
```

---

## Trajectory Analysis

### What is a Trajectory?

A trajectory records RLM's reasoning process:
- Queries and sub-queries
- REPL code executed
- Results at each step
- Final answer synthesis

### Verbosity Levels

| Level | Shows |
|-------|-------|
| `minimal` | RECURSE, FINAL, ERROR only |
| `normal` | All events, truncated content |
| `verbose` | All events, full content |
| `debug` | Everything + internal state |

### Analyzing Trajectories

```
/trajectory ~/.claude/trajectories/session-123.json
```

Output:
```
Trajectory Analysis
───────────────────
Total events: 23
Max depth reached: 2
Recursive calls: 4
REPL executions: 8
Duration: 34.2s
Estimated cost: $0.47

Event Distribution:
  ANALYZE: 3
  REPL_EXEC: 8
  RECURSE_START: 4
  RECURSE_END: 4
  FINAL: 1
```

---

## Strategy Learning

RLM learns from successful trajectories.

### Strategy Types

| Strategy | Description | When Used |
|----------|-------------|-----------|
| Peeking | Sample context before deep dive | Large files, unknown structure |
| Grepping | Pattern-based search | Finding specific code patterns |
| Partition+Map | Divide and conquer | Multi-file analysis |
| Programmatic | One-shot code execution | Transformations, calculations |
| Recursive | Spawn sub-queries | Verification, complex reasoning |

### How Learning Works

1. **Pattern Detection**: Identifies strategies used in successful trajectories
2. **Feature Extraction**: Extracts query characteristics
3. **Similarity Matching**: Matches new queries to past successes
4. **Strategy Suggestion**: Suggests proven approaches

### Viewing Strategy Suggestions

With debug verbosity:
```
[STRATEGY] Similar query found (similarity: 0.89)
  Previous: "Find all TODO comments in src/"
  Strategy: grepping (effectiveness: 0.94)
  Suggestion: Use search() with regex pattern
```

---

## Advanced Configuration

### Full Config File

`~/.claude/rlm-config.json`:

```json
{
  "activation": {
    "mode": "complexity",
    "fallback_token_threshold": 80000,
    "auto_activate": true,
    "complexity_threshold": 0.6
  },
  "depth": {
    "default": 2,
    "max": 3
  },
  "models": {
    "root_model": "opus",
    "recursive_depth_1": "sonnet",
    "recursive_depth_2": "haiku",
    "prefer_provider": "anthropic"
  },
  "trajectory": {
    "verbosity": "normal",
    "streaming": true,
    "save_to_disk": true,
    "save_path": "~/.claude/trajectories"
  },
  "cost": {
    "session_budget": 5.0,
    "warn_at_percent": 80
  },
  "tools": {
    "default_access": "read_only",
    "blocked_commands": ["rm -rf", "sudo"]
  }
}
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API access |
| `OPENAI_API_KEY` | OpenAI API access (optional) |
| `RLM_CONFIG_PATH` | Custom config location |
| `RLM_DEBUG` | Enable debug logging |

---

## Best Practices

### 1. Start with Balanced Mode

The default balanced mode works well for most tasks. Only switch to thorough for genuinely complex work.

### 2. Use Budgets

Set a reasonable budget to prevent unexpected costs:
```
/rlm budget $2
```

### 3. Review Trajectories for Complex Tasks

For important decisions, check the trajectory to understand RLM's reasoning:
```
/rlm verbosity verbose
```

### 4. Use /simple for Quick Questions

Don't waste RLM overhead on simple queries:
```
/simple
What's the syntax for a Python list comprehension?
```

### 5. Leverage Memory for Recurring Work

Store facts about your codebase to improve future sessions:
```python
memory_add_fact("This project uses FastAPI with SQLAlchemy", confidence=0.95)
```

### 6. Provide Context in Queries

Help RLM make better decisions:
```
# Good - clear scope
"Analyze the authentication flow in src/auth/"

# Less good - vague
"Check the code"
```

### 7. Use Thorough Mode for Security

For security-sensitive work:
```
/rlm mode thorough
Find security vulnerabilities in the payment processing code
```

---

## Getting Help

- **GitHub Issues**: [github.com/rand/rlm-claude-code/issues](https://github.com/rand/rlm-claude-code/issues)
- **Getting Started**: [getting-started.md](./getting-started.md)
- **Specification**: [rlm-claude-code-spec.md](../rlm-claude-code-spec.md)
