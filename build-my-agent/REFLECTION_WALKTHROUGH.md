# Phase 5: What We've Built — System Map

## The 10-Module Architecture

### Foundation Layer
| Module | File | Responsibility |
|--------|------|----------------|
| 1 | `state.py` | `AgentState` dataclass — the single source of truth for any agent run |
| 2 | `config.py` | LLM connection (vLLM endpoint, model name, client singleton) |
| 3 | `parser.py` | 4-strategy fallback (direct JSON, markdown, braces, keywords) + None-safe |

### Core Agent Layer
| Module | File | Agent Type | Control Flow |
|--------|------|-------------|--------------|
| 4 | `agent_loop.py` | Simple Loop | While: think → act/answer → observe |
| 5 | `react_agent.py` | ReAct | Structured Thought → Action → Observation cycles |
| 6 | `plan_agent.py` | Plan-and-Execute | Plan → Execute sub-tasks with dependency resolution → Synthesize |
| 7 | `reflection_agent.py` | Reflection | Critique → Score → Revise → Re-critique (wraps any of above) |

### Support Layer
| Module | File | Responsibility |
|--------|------|----------------|
| 8 | `memory.py` | Sliding window, protected messages, token-aware trimming |
| 9 | `tools.py` | 5 tools (Python, read/write file, search KB, calculator) with structured schemas |
| 10 | `guard.py` | 3-layer defense: Name Resolution → Semantic Recovery → Schema Validation |
| 11 | `knowledge_base.py` | 23-fact dictionary with keyword search and relevance scoring |

## The Reflection Agent — Deep Dive

### The Core Idea

The first 4 agent types (Simple Loop, ReAct, Plan-and-Execute) all share a fundamental limitation:

**They produce an answer and return it. No one checks if the answer is actually good.**

Reflection adds a **quality assurance layer**. After the inner agent finishes, a separate "critic" evaluates the answer, and if it's below a quality threshold, a "reviser" fixes it. The cycle repeats until the answer passes or we hit the maximum attempts.

```
         ┌──────────────────────────────────────────┐
         │  ReflectionAgent.run(goal)                │
         │                                           │
         │  Phase 1: inner_agent.run(goal)           │
         │            │                               │
         │            ▼ first draft answer            │
         │     ┌────────────────┐                    │
         │     │ Critique       │                    │
         │     └────────┬───────┘                    │
         │              │                             │
         │     ┌────────▼───────┐                   │
         │     │  score >= 7.0?  │── Yes ──→ ACCEPT  │
         │     └────────┬───────┘                    │
         │     No       │                             │
         │              ▼                             │
         │     ┌────────────────┐                    │
         │     │ Revise         │                    │
         │     └────────┬───────┘                    │
         │              │ (loop back)                 │
         └──────────────┴───────────────────────────┘
```

### The Data Structures

**One specific issue found by the critic:**
```python
@dataclass
class Critique:
    category: str    # "accuracy" | "completeness" | "clarity" | "format"
    issue: str       # What's wrong
    suggestion: str  # How to fix it
```

Example: `Critique(category="completeness", issue="Only 2 of 3 topics covered", suggestion="Add the 3rd topic about light-seconds")`

**The full evaluation report:**
```python
@dataclass
class CritiqueReport:
    score: float         # Overall 1-10 (average of 4 dimensions)
    scores: dict         # {"accuracy": 8, "completeness": 4, "clarity": 7, "format": 6}
    critiques: list      # [Critique, Critique, ...]
    is_acceptable: bool  # True if score >= threshold
```

**Why all three levels instead of one number?**

- **Overall score**: The go/no-go decision (enough to control the loop)
- **Dimension scores**: Show the reviser *where* the weaknesses are (completeness is the problem, not clarity)
- **Specific critiques**: Actionable items the reviser can fix one by one

### The Critic (Parts 2-3)

The critic is a separate LLM call that evaluates the answer. Its prompt includes:
1. The original goal (what the user wanted)
2. The answer to evaluate (the first draft)
3. Instructions for 4-dimension scoring
4. A request for specific issues with suggestions

**The pragmatic reality:** This LLM often returns natural language instead of strict JSON. The `_parse_natural_response()` fallback extracts:
- Overall score (regex: `Score: X/10`)
- Dimension scores (regex: `accuracy: 7/10`)
- Issues (lines containing "missing", "incomplete", "incorrect", "lacks")

This is **defensive programming** — the system works even when the LLM doesn't follow instructions perfectly.

**Temperature=0.3 for the critic** means deterministic evaluation. Same answer should get same score.

### The Reviser (Parts 4-5)

The reviser is a separate LLM call that improves the answer. Its prompt includes:
1. The original goal
2. The current flawed answer
3. The critique report (overall score, dimension scores, specific issues)
4. Instructions to fix ALL issues
5. Request for only the revised answer (no preamble)

**Temperature=0.7 for the reviser** allows creative rewriting. We want natural, improved prose — not rigid formula-filling.

The reviser is intentionally simple (one LLM call, not an agent loop) because its job is just "rewrite based on feedback." It doesn't need to run tools or search the knowledge base.

### The Reflection Agent (Part 6)

This is the **orchestrator** that ties the critic and reviser into a loop.

**Constructor:**
```python
class ReflectionAgent:
    def __init__(self, inner_agent, threshold=7.0, max_iterations=3, model=None):
```

**The `run()` method has 3 phases:**

**Phase 1: Get the first draft**
- Call `inner_agent.run(goal)` (could be Plan-and-Execute, ReAct, or simple loop)
- Extract the first-draft answer

**Phase 2: Reflection loop**
```
for iteration in 1..max_iterations:
    report = critic evaluates current_answer
    history.append(iteration, score, pass/fail, num_issues)
    
    if report.is_acceptable:
        break  # Done — the answer meets the quality bar
    
    current_answer = reviser fixes the issues
```

**Phase 3: Return the result**
```python
return {
    "goal": goal,
    "initial_answer": first draft,
    "final_answer": polished version,
    "reflection_history": journey (each iteration's score and status),
    "iterations": how many cycles,
    "threshold": the quality bar,
    "inner_agent_result": full details from the inner agent,
}
```

**The two exit conditions:**
1. **Early exit (ideal):** Score meets threshold on some iteration — loop breaks, no more revising
2. **Max iterations (safety valve):** We've tried max_iterations times without reaching the threshold — return the best attempt

The safety valve is critical. Without it, a pathological case (reviser makes things worse each time) would loop forever.

## Validation Results

Our LLM tests confirmed the system works:

**Test 1 — Critic with intentional flaws:**
- Input: 2 facts, no structure, missing section 3
- Result: 5.0/10, flagged completeness issue ✓

**Test 2 — Reviser fixes the flaws:**
- Input: The 5.0/10 answer + critique feedback
- Result: 267-char answer with 3 sections, bullet points, all topics covered ✓

**Test 4 — Full reflection loop:**
```
Iteration 1: 5.0/10 (FAIL) → Reviser produces complete 3-section report
Iteration 2: 10.0/10 (PASS) → Loop breaks, answer accepted

Score trajectory: 5.0 → 10.0 (+5.0 improvement)
```

The reflection loop successfully transformed a flawed first draft into a high-quality final answer.

---

## Where We Are

You have built a **10-module AI agent system from scratch**. Every line of code was written, tested, and understood. Here's the progression:

```
Module 1-2:  Basic building blocks (state, LLM connection, parsing)
Module 3-4:  Simple agent loop (the foundational "while" pattern)
Module 5-6:  Memory and tools (give the agent capabilities)
Module 7:     Guard (3-layer defense against LLM errors)
Module 8:     ReAct (structured reasoning with visible thought traces)
Module 9:     Plan-and-Execute (proactive planning before action)
Module 10:    Reflection (post-execution quality assurance)
```

Each module solved a real problem:
- No tools? → Added tools (code execution, file I/O, search)
- No reasoning? → Added ReAct (Thought → Action → Observation)
- No planning? → Added Plan-and-Execute (plan first, execute in order)
- No quality check? → Added Reflection (critique → revise → re-critique)

## What's Next — Choose Your Path

You've completed the core curriculum. Here are the next directions:

### A) Production Hardening
Make the system robust for real-world use:
- Structured logging (log every step, not just print statements)
- Token usage tracking (how many tokens per step, total cost)
- Timeout handling and circuit breakers
- Integration test suite

### B) Long-Term Memory
Add persistent memory that survives across sessions:
- Embedding vectors (convert text to numbers)
- Vector similarity search (find related memories)
- Memory consolidation (compress old memories into summaries)

### C) Multi-Agent Systems
Run multiple specialized agents that collaborate:
- One agent plans, another executes, another reviews
- Routing (which agent handles which type of task)
- Consensus (multiple agents evaluate, majority wins)

### D) Advanced Tool Learning
Teach the agent to discover and use new tools:
- Tool description parsing (agent reads a new tool's docstring)
- Dynamic tool registration (add tools at runtime)
- Tool composition (chain tools together)

### E) Capstone Project
Build a complete application using everything you've learned:
- A research assistant (search, analyze, write report, self-review)
- A code review bot (read code, find issues, suggest fixes, verify)
- A tutoring system (teach a topic, check understanding, adapt)

**Which path interests you most?**
