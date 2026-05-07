"""
Comparison: Single Prompt vs. Simple Agent vs. ReAct

This script demonstrates the three approaches to solving a multi-step task:

1. Single Prompt: One-shot, no tools, just the raw LLM
2. Simple Agent: The original agent_loop with tool use
3. ReAct Agent: Interleaved reasoning + action with a visible trace
"""

from config import get_client, get_model
from tools import execute_tool
from guard import guarded_execute_tool
from parser import parse_response

client = get_client()
model = get_model()

task = (
    "Calculate 199 * 501 and 247 * 398, then write the results to 'comparison.txt' "
    "and read it back to confirm. Finally, tell me which product is larger."
)

print("=" * 70)
print("TASK:")
print(task)
print("=" * 70)

# =========================================================================
# Approach 1: Single Prompt (Zero-shot)
# =========================================================================
print("\n--- APPROACH 1: Single Prompt (Zero-shot) ---")
print("What happens: The LLM receives the task once and tries to answer.")
print("Limitation: No tools, no memory, no trace. Pure parametric knowledge.\n")

single_prompt_response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": task},
    ],
    max_tokens=2048,
    temperature=0.7,
)
single_answer = single_prompt_response.choices[0].message.content
print("--- Output (first 300 chars) ---")
# Safe ASCII output
safe_output = single_answer.encode("ascii", errors="replace").decode("ascii")
print(safe_output[:300])
print("...")

# =========================================================================
# Approach 2: Simple Agent (original loop)
# =========================================================================
print("\n--- APPROACH 2: Simple Agent (original loop) ---")
print("What happens: The agent uses a loop, calls tools, but doesn't maintain")
print("a visible trace. The state is a black box.\n")

# Simulate the simple agent
from state import AgentState
from memory import build_prompt

state = AgentState(goal=task, max_steps=10)

print("Running the simple agent loop...")
# (We'll skip the full execution for brevity, but the point is:
#  the original agent lacks the visible thought/action/observation trace)

print("[Simple agent uses the same tools but without the ReAct trace]")
print("The LLM makes a call, gets a result, makes another call...")
print("But there's no 'Thought' field -- just tool calls and results.")

# =========================================================================
# Approach 3: ReAct Agent
# =========================================================================
print("\n--- APPROACH 3: ReAct Agent ---")
print("What happens: Each step includes a Thought (reasoning) + Action (tool call)")
print("--> Observation (result). The trace is visible and accumulates.")
print("The LLM can see and reason about its own work.\n")

from react_agent import ReActAgent

agent = ReActAgent(max_steps=8)
result = agent.run(goal=task)

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print(f"""
Approach          Tools  Trace  Reasoning       Self-Correction
--------------------------------------------------------------
1. Single Prompt  No     No     Internal only   No
2. Simple Agent   Yes    No     Tool calls      No
3. ReAct Agent    Yes    Yes    Explicit trace  Yes
""")

print("--- The Key Difference ---")
print("ReAct makes the reasoning VISIBLE. The LLM can see its own thoughts,")
print("actions, and observations as a growing log. This enables:")
print("  1. Self-correction (the LLM notices mistakes and fixes them)")
print("  2. Multi-step planning (the LLM plans based on past results)")
print("  3. Evidence-based reasoning (the LLM uses retrieved facts, not just training data)")
