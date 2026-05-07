# ReAct Agent — Build Summary

## What is ReAct?

ReAct (Reasoning + Acting) is an agentic pattern that **interleaves** reasoning and action. Instead of the LLM thinking first then acting (or vice versa), it alternates:

```
Thought → Action → Observation → Thought → Action → ... → Answer
```

The **key insight**: the trace becomes part of the conversation history. The LLM sees its own reasoning and can course-correct.

## Architecture

### 1. ReAct Cycle (`ReactCycle`)
The atomic unit of the loop:
```python
@dataclass
class ReactCycle:
    thought: str          # The LLM's reasoning
    action: str           # JSON of the tool call
    observation: Optional  # Filled after execution
```

### 2. ReAct State (`ReActState`)
Tracks the agent's progress:
- Goal
- List of completed cycles (the trace)
- Current step / max steps
- Final answer + completion flag

### 3. System Prompt
Forces the LLM to:
- Always include a `thought` field explaining reasoning
- Use one of: `use_tool` (with thought), `answer` (with thought), or `think` (planning)
- Read observations and reason about them

### 4. The Agent Loop
```
1. Build messages = [system prompt, goal, all previous cycles as trace]
2. Call LLM with the trace
3. Parse response (extracts thought + action)
4. If use_tool: execute via guarded pipeline, record observation
5. If answer: terminate with final answer
6. Loop until done or max_steps
```

## What We Modified

### `parser.py`
- `normalize_action()` now passes through the `thought` field when present
- Maintains backward compatibility (no thought = no key in output)

### `react_agent.py` (NEW)
- `ReActAgent` class with full ReAct loop
- `_build_messages()`: converts cycles to conversation history
- `_call_llm()`: calls the LLM with trace context
- `step()`: executes one ReAct cycle
- `run()`: the main entry point

## Test Results

**Task**: Calculate 199*501 + 247*398, write to file, read back, answer

**Result**: 4 steps, 3 cycles, 16.8s
```
Cycle 1: "I need to calculate... Let me compute this first."
  → python → 198005

Cycle 2: "The result is 198005. Now I need to write this..."
  → write_file → Success

Cycle 3: "The file was written. Now I need to read it back..."
  → read_file → 198005

Answer: "The final sum is 198005. This has been confirmed."
```

The agent demonstrated **self-correction**, **multi-step planning**, and **evidence-based reasoning** — all hallmarks of the ReAct pattern.

## Why This Matters

Compared to a simple agent loop:
- **Single Prompt**: LLM guesses, no tools, hallucinates
- **Agent Loop**: LLM plans → executes, but can't see its work
- **ReAct**: LLM plans → executes → **sees result** → adjusts plan → executes → concludes

The trace is the difference. It's the mechanism that enables complex, self-correcting behavior.
