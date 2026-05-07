# Agent Architecture Comparison

## The Evolution of Agent Design

This document compares 4 agent architectures, showing how each builds on the previous one.

---

## 1. Single Prompt (Baseline)

### How it works
- One LLM call with the full prompt
- The LLM either answers directly or admits it can't

### Code (conceptual)
```python
def single_prompt(goal):
    # Just one call, no loop, no tools
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": goal}]
    )
    return response.choices[0].message.content
```

### Strengths
- Fastest (one call)
- Simplest to understand
- No loops, no error handling

### Weaknesses
- Cannot use tools
- Cannot verify its own answers
- Hallucinates when it doesn't know
- Context window limits complex tasks

### Example
Goal: "Calculate 987 * 654 and write to a file"

Response: "I can't execute calculations or write files. 987 * 654 = 645,198"
(The LLM tries to calculate in its head, but can't write the file)

---

## 2. Simple Agent Loop

### How it works
- Loop: Think → Decide → Act → Observe → Repeat
- Can use tools in the loop
- No structured reasoning trace

### Code (conceptual)
```python
while not done:
    response = client.chat.completions.create(
        messages=history + [user_message]
    )
    action = parse(response)
    if action == "answer":
        break
    result = execute_tool(action)
    history.append(result)
```

### Strengths
- Can use tools
- Can iterate until complete
- Can handle multi-step tasks

### Weaknesses
- No visible reasoning (black box)
- No self-correction
- Can loop forever
- No lookahead (myopic)

### Example
Goal: "Calculate 987 * 654 and write to a file"

Step 1: Call python tool with "987 * 654"
Step 2: Call write_file with the result
Step 3: Return answer

(Works, but the agent is just reacting to each step, not planning ahead)

---

## 3. ReAct Agent

### How it works
- Interleaved Thought → Action → Observation cycles
- The thought is recorded in a structured trace
- The trace is fed back to the LLM on each step

### Code (conceptual)
```python
while not done:
    thought = "What should I do next?"
    action = call_llm(trace + thought)
    observation = execute(action)
    trace.append(ReactCycle(thought, action, observation))
```

### Strengths
- **Visible reasoning** - The thought trace shows the agent's logic
- **Self-correction** - Can see mistakes and adjust
- **Structured** - Cycles are recorded, not just raw messages

### Weaknesses
- **Myopic** - Only sees the next step, no plan
- **Context drift** - Long traces can confuse the LLM
- **Over-thinking** - May loop unnecessarily
- **No commitment** - Doesn't plan the end, just reacts

### Example
Goal: "Calculate 987 * 654 and write to a file"

Cycle 1:
- Thought: "I need to calculate the product first"
- Action: Call python with "987 * 654"
- Observation: "645198"

Cycle 2:
- Thought: "Now I need to write this to a file"
- Action: Call write_file with "645198"
- Observation: "File written"

Cycle 3:
- Thought: "I have the result and file, I can answer"
- Action: Answer
- Result: "The product is 645,198, written to file"

(Works well for simple tasks, but struggles with complex multi-part goals)

---

## 4. Plan-and-Execute Agent (THE BREAKTHROUGH)

### How it works
- **Phase 1 (Plan):** LLM creates a full plan with numbered steps and dependencies
- **Phase 2 (Execute):** Each step is handled by a focused ReAct sub-agent
- **Phase 3 (Synthesize):** Final assembly of all partial results

### Code (conceptual)
```python
plan = create_plan(goal)  # One LLM call to plan

results = {}
for step in plan.steps:
    sub_goal = build_sub_goal(goal, plan, step, results)
    sub_agent = ReActAgent(goal=sub_goal)
    results[step.number] = sub_agent.run()

final = synthesize(goal, plan, results)  # One LLM call to assemble
```

### Strengths (vs. ReAct)
- **Proactive, not reactive** - The plan is made BEFORE execution
- **Focused sub-tasks** - Each step has a clear, independent goal
- **Dependency management** - Steps can run in any order (as long as deps are met)
- **Error isolation** - If one sub-task fails, others can still complete
- **Context efficiency** - Each sub-agent only sees what it needs

### Weaknesses
- **Overhead** - Planning phase costs 1 extra LLM call
- **Over-execution** - Sub-agents may do more work than planned
- **No re-planning** - If the plan is wrong, the whole thing fails
- **Max steps** - Sub-task limits can cut off complex steps

### Example
Goal: "Research: speed of light, capital of France, calculate light-years, write to file"

**Plan (created before execution):**
- Step 1: Search for speed of light (no deps)
- Step 2: Search for capital of France (no deps)
- Step 3: Calculate light-year distance (deps: 1, 2)
- Step 4: Write to file (deps: 1, 2, 3)

**Execution (step by step):**
- Step 1 runs (search) → Result: "299,792,458 m/s"
- Step 2 runs (search) → Result: "Paris"
- Step 3 runs (calculate) → Uses results from 1 & 2
- Step 4 runs (write) → Uses results from 1, 2, & 3

**Synthesis (final assembly):**
Combines all 4 results into a coherent report

---

## Comparison Table

| Feature | Single Prompt | Simple Loop | ReAct | Plan-and-Execute |
|---------|--------------|-------------|-------|-------------------|
| Tool Use | ❌ | ✅ | ✅ | ✅ (per sub-task) |
| Visible Reasoning | ❌ | ❌ | ✅ (trace) | ✅ (plan + trace) |
| Self-Correction | ❌ | ❌ | ✅ (in trace) | ✅ (per sub-task) |
| Lookahead / Planning | ❌ | ❌ | ❌ (myopic) | ✅ (full plan) |
| Dependency Management | ❌ | ❌ | ❌ (implicit) | ✅ (explicit) |
| Error Isolation | ❌ | ❌ | ❌ (cascades) | ✅ (independent) |
| Context Efficiency | N/A | ❌ (grows) | ❌ (full trace) | ✅ (focused) |
| Max Steps | N/A | Fixed | Fixed | Per sub-task |
| Extra LLM Calls | 1 | 1 per step | 1 per cycle | 1 (plan) + 1 (synthesize) + sub-tasks |

---

## When to Use Each

- **Single Prompt** → Simple Q&A, no tools needed
- **Simple Loop** → Quick iterations, no reasoning trace needed
- **ReAct** → Complex single tasks, self-correction needed
- **Plan-and-Execute** → Multi-part goals, many sub-tasks, need coordination

## What's Next?

The next logical evolution: **Notebook 07 - Reflection and Self-Critique**

After the Plan-and-Execute agent completes, we can add a **post-execution review phase**:
- Critique the final answer against a rubric
- Score quality, completeness, accuracy
- Revise if needed
- This catches errors that the plan didn't foresee

This is the difference between:
- "I followed the plan and got an answer" (Plan-and-Execute)
- "I followed the plan, got an answer, and verified it meets the requirements" (Reflection)
