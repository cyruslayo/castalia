# Notebook 06: Plan and Execute - Summary

## What We Built

Created a Plan-and-Execute agent that adds a **pre-execution planning phase** on top of the ReAct agent.

## Architecture

The system has 3 phases:

### Phase 1: Planning
- The **Planner** is a single LLM call (not a loop)
- It receives the user's goal and the list of available tools
- It returns a structured `Plan` with numbered `PlanStep` objects
- Each step has: number, description, and dependencies
- Dependencies encode which steps must complete before another can start

### Phase 2: Execution
- For each step (in dependency order):
  1. Build a focused sub-goal with context from previous steps
  2. Run a **ReActAgent** to handle that sub-goal
  3. Record the result in a `results` dict keyed by step number

The execution loop uses **topological sort** to respect dependencies:
1. Find all "ready" steps (all dependencies satisfied)
2. Execute the first ready step
3. Mark it as completed
4. Repeat until all steps are done

### Phase 3: Synthesis
- A final LLM call combines all step results into a coherent answer
- The synthesizer sees: original goal, plan summary, and all partial results

## Key Data Structures

```
Plan
  summary: str          # One-line overview
  steps: list[PlanStep] # Ordered steps with dependencies

PlanStep
  number: int           # Unique ID
  description: str      # What to do in this step
  dependencies: list    # Which other steps must finish first
```

## Critical Implementation Details

### The JSON Parser Problem
The original code used `parse_response()` to read the planner's output. This was wrong because:
- `parse_response()` expects agent actions (action/thought/answer fields)
- The planner returns a different schema (summary/steps fields)
- The mismatch caused the data to be treated as a "thought" string instead of structured data

**Solution:** Wrote `_extract_json()` - a simple parser that:
- Handles `None` defensively
- Finds the first `{` and last `}`
- Parses with `json.loads()`
- Returns empty dict on failure

### Temperature Settings
- **Planner:** `temperature=0.3` (deterministic, consistent plans)
- **Synthesizer:** `temperature=0.7` (creative, natural language final answer)

### Sub-task Size
- Default: 8 steps per sub-task (configurable)
- Can be increased for more complex individual steps

## Test Results

Test Goal: "Research: speed of light, capital of France, calculate light-years, write to report.txt"

Generated Plan (4 steps):
1. Search knowledge base for speed of light (no deps)
2. Search knowledge base for capital of France (no deps)
3. Calculate light-year distance (depends on 1, 2)
4. Write results to report.txt (depends on 1, 2, 3)

Execution Results:
- Step 1: 6 steps, 52s (searched + did extra work)
- Step 2: 2 steps, 7s (clean search)
- Step 3: 2 steps, 29s (python calculation)
- Step 4: 2 steps, 17s (write file)
- Total: 12 steps, 105s

Final synthesis produced a coherent, well-structured answer combining all results.

## Files Modified

- `plan_agent.py` (NEW, 304 lines) - Complete plan-and-execute agent
- `react_agent.py` (MODIFIED) - Added Unicode-safe print for Windows console
- `guard.py` - No changes
- `tools.py` - No changes

## Curriculum Alignment

| Notebook | Concept in Our Build |
|----------|----------------------|
| 06 (Plan and Execute) | Core planning + execution with dependency resolution |
| 05 (ReAct) | Sub-tasks are executed by ReAct agents |
| 03 (Tool Use) | Planner knows what tools are available |
| 04 (Structured Parsing) | Custom JSON extractor for planner output |
| 24 (Guardrails) | Guard protects all sub-task tool calls |

## Pedagogical Takeaway

The Plan-and-Execute architecture demonstrates **separation of concerns**:
- The **Planner** handles strategy (what to do, in what order)
- The **ReAct executor** handles tactics (how to do each sub-task)
- The **Synthesizer** handles communication (how to present the answer)

This is the same pattern used in production systems like LangChain's `PlanAndExecute` and AutoGPT's task decomposition.
