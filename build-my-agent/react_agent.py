"""
ReAct Agent — The Foundational Agentic Pattern
================================================

ReAct (Reasoning + Acting) was introduced by Yao et al. (2022).
The key insight: instead of having the LLM think THEN act, or act THEN think,
ReAct INTERLEAVES them.

Each step produces:
  1. Thought — the LLM's reasoning about what to do
  2. Action — a specific tool call with parameters
  3. Observation — the result from the tool, fed back to the LLM

The trace (Thought→Action→Observation) becomes part of the conversation history,
allowing the LLM to reason about its own reasoning.

Build on existing modules:
  - tools.py (ToolRegistry, execute_tool, etc.)
  - parser.py (parse_response for structured output)
  - guard.py (3-layer guard: name resolution → recovery → validation)
  - config.py (LLM client)
  - memory.py (build_prompt for context management)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

# Import our existing modules
from tools import generate_tools_instruction, execute_tool
from guard import guarded_execute_tool  # Our 3-layer safety net
from parser import parse_response
from config import get_client, get_model
from memory import build_prompt


# ====================================================================
# Part 1: The ReAct Trace
# ====================================================================
#
# A ReAct trace is a log of the agent's reasoning process.
# Each cycle has: Thought, Action, Observation
#
# The trace is what makes ReAct special — it's visible to the LLM,
# allowing it to see and continue its own reasoning.

@dataclass
class ReactCycle:
    """
    One cycle in the ReAct trace: Thought → Action → Observation

    This is the atomic unit of ReAct. The LLM produces the Thought and Action,
    we execute the Action, and the result becomes the Observation.
    """
    thought: str
    action: str  # JSON-formatted tool call
    observation: Optional[str] = None  # Filled in after execution


@dataclass
class ReActState:
    """
    State for the ReAct agent.

    We could reuse AgentState, but ReAct has a distinct pattern:
    - It maintains a visible trace of cycles
    - The trace IS the memory (no separate state dict)
    - It terminates when the LLM produces an "Answer" instead of "Action"
    """
    goal: str
    cycles: list = field(default_factory=list)  # List of ReactCycle
    current_step: int = 0
    max_steps: int = 10  # Higher limit for complex chains
    is_complete: bool = False
    final_answer: Optional[str] = None
    start_time: float = field(default_factory=time.time)

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> str:
        lines = [
            f"Goal: {self.goal}",
            f"Steps: {self.current_step}/{self.max_steps}",
            f"Time: {self.elapsed_time():.1f}s",
            f"Cycles completed: {len(self.cycles)}",
        ]
        if self.is_complete:
            lines.append(f"Final answer: {self.final_answer}")
        return " | ".join(lines)


# ====================================================================
# Part 2: The ReAct System Prompt
# ====================================================================
#
# This is the prompt that tells the LLM HOW to behave as a ReAct agent.
# The format is critical — the LLM must know to output Thought/Action/Answer.

def build_react_system_prompt(tools_instruction: str) -> str:
    """
    Build the system prompt for the ReAct agent.

    The key design choices:
    1. Every action MUST include a "thought" field explaining the reasoning.
       This forces the LLM to interleave reasoning with action (the ReAct pattern).

    2. The three actions are: use_tool, answer, think
       - use_tool: includes thought + tool call (primary action mode)
       - answer: final conclusion (termination signal)
       - think: pure reasoning (rare, for initial analysis)

    3. The observation from a tool call is fed back as a message,
       and the NEXT response must reason about it in the thought field.

    4. The prompt includes the tools instruction so the LLM knows
       what tools are available and how to call them.
    """
    return f"""You are a ReAct (Reasoning + Acting) agent. You solve problems by interleaving reasoning and action.

## The ReAct Pattern

For each turn, you will:
  1. Read the current situation (your previous observations + the goal)
  2. Think about what to do next
  3. Either call a tool OR give the final answer

## How to respond (output format)

ALWAYS output a single JSON object. Every response must include a "thought" field
explaining your reasoning (what you understand, what you plan to do, and why).

### To use a tool (most common response):

{{
  "action": "use_tool",
  "thought": "I need to [goal]. I'll use the [tool] to [reason for choosing this tool].",
  "tool": "tool_name",
  "params": {{...}}
}}

### To give the final answer (when you have enough information):

{{
  "action": "answer",
  "thought": "I have [summarize what I found]. I can now answer the question.",
  "answer": "Your complete answer here."
}}

### To think (rarely needed, for initial planning only):

{{
  "action": "think",
  "thought": "My reasoning here..."
}}

## Critical Rules

- ALWAYS include the "thought" field in your response. Explain your reasoning clearly.
- After each tool call, the system will show you an Observation. READ IT and reason about it in your next thought.
- Keep your thoughts concise but informative. Show your work.
- Only use "answer" when you genuinely have enough information to conclude.

{tools_instruction}

Now, reason about the task below and take your first step.
"""


# ====================================================================
# Part 3: The ReActAgent Class
# ====================================================================
#
# This is the core engine. It manages the ReAct loop, maintains
# the trace, and integrates with all our existing infrastructure.

class ReActAgent:
    """
    A complete ReAct agent that interleaves reasoning and action.

    The agent:
    1. Receives a goal/task
    2. Enters a loop: Thought → Action → Observation
    3. Terminates when it produces an Answer or hits max_steps
    4. Returns the final answer + the full trace

    It uses our existing:
    - Guard system (3-layer: name resolution, recovery, validation)
    - Parser (4-strategy fallback)
    - Memory (context management)
    """

    def __init__(self, max_steps: int = 10, model: Optional[str] = None):
        self.state = ReActState(goal="", max_steps=max_steps)
        self.model = model or get_model()

    def _build_messages(self, goal: str) -> list:
        """
        Build the conversation for the current step.

        This is where the trace becomes part of the prompt:
        - System message with ReAct instructions
        - The original goal
        - All previous cycles as messages (the trace)

        The LLM sees its own reasoning history and can continue from there.
        """
        tools_instruction = generate_tools_instruction()
        system_prompt = build_react_system_prompt(tools_instruction)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]

        # Add the trace as assistant/observation messages
        for cycle in self.state.cycles:
            # The agent's thought and action (as one assistant message)
            assistant_content = f"Thought: {cycle.thought}\n"
            assistant_content += f"Action: {cycle.action}"
            messages.append({"role": "assistant", "content": assistant_content})

            # The observation (as a user/system message so the LLM can read it)
            if cycle.observation is not None:
                messages.append({
                    "role": "user",  # Some implementations use "system" for observations
                    "content": f"Observation: {cycle.observation}"
                })

        return messages

    def _call_llm(self, messages: list) -> str:
        """Call the LLM and get raw response text."""
        client = get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def _execute_action(self, action_data: dict) -> Optional[str]:
        """
        Execute a tool action and return the observation.

        If the action is "use_tool", we call the guarded tool executor.
        If it's "think", there's no execution (just reasoning).
        If it's "answer", the task is complete.
        """
        action_type = action_data.get("action")

        if action_type == "think":
            return None  # No execution for thinking

        if action_type == "answer":
            return None  # No execution for final answer

        if action_type == "use_tool":
            tool_name = action_data.get("tool", "")
            params = action_data.get("params", {})

            # Use our 3-layer guarded executor
            result = guarded_execute_tool(tool_name, params)
            return result

        return "Observation: Unknown action type"

    def step(self) -> Optional[str]:
        """
        Execute one step of the ReAct loop.

        1. Build messages (including the trace)
        2. Call the LLM
        3. Parse the response into a Thought and/or Action
        4. If it's a tool action, execute it and record the observation
        5. If it's an answer, terminate

        Returns:
            "continue" if the loop should continue
            "done" if the task is complete (answer produced)
            "max_steps" if we hit the limit
        """
        if self.state.current_step >= self.state.max_steps:
            return "max_steps"

        self.state.current_step += 1

        # Build the conversation with the current trace
        messages = self._build_messages(self.state.goal)

        # Get the LLM's response
        raw_response = self._call_llm(messages)

        # Parse the response (using our 4-strategy fallback parser)
        parsed = parse_response(raw_response)

        action_type = parsed.get("action")

        # Handle different action types
        if action_type == "think":
            # Pure reasoning step — record the thought but don't add to cycles yet
            # The agent is planning; it will act in the next step with the trace intact
            thought = parsed.get("thought", "No thought provided")
            print(f"  Step {self.state.current_step} [THINK]: {thought[:80]}...")
            # Don't add a cycle; let the next action carry the planning forward

        elif action_type == "use_tool":
            # Action step — execute the tool with reasoning
            tool_name = parsed.get("tool", "unknown")

            # Extract the thought (the LLM sends it in the JSON, parse_response may not extract it)
            thought = parsed.get("thought", "")
            if not thought:
                # Fallback: the parser might have stripped it, check raw response
                thought = f"Using {tool_name} to make progress on the task."

            cycle = ReactCycle(
                thought=thought,
                action=json.dumps(parsed),
            )

            # Execute the action
            observation = self._execute_action(parsed)
            cycle.observation = observation

            print(f"  Step {self.state.current_step} [ACT: {tool_name}]")
            print(f"  [OBSERVATION]: {str(observation)[:100]}...")

            # Add to the trace
            self.state.cycles.append(cycle)

        elif action_type == "answer":
            # Termination — the agent has a final answer
            answer = parsed.get("answer", "No answer provided")
            self.state.final_answer = answer
            self.state.is_complete = True
            print(f"  Step {self.state.current_step} [ANSWER]: {answer[:80]}...")
            return "done"

        else:
            # Unknown action — record as a cycle with a warning
            cycle = ReactCycle(
                thought=f"Received unrecognized response: {str(parsed)[:100]}",
                action="",
                observation="Observation: The system couldn't parse your response. Please use the correct format."
            )
            self.state.cycles.append(cycle)

        return "continue"

    def run(self, goal: str) -> dict:
        """
        Run the ReAct agent on a given goal.

        This is the main entry point. The agent will:
        1. Set the goal
        2. Enter the ReAct loop
        3. Stop when it produces an answer or hits max_steps
        4. Return the final state

        Args:
            goal: The task or question to solve

        Returns:
            A dict with:
            - "answer": the final answer (if complete)
            - "cycles": the trace of thought/action/observation cycles
            - "is_complete": whether the agent finished successfully
            - "steps": number of steps taken
            - "time": total time in seconds
        """
        self.state.goal = goal
        print("=" * 60)
        print(f"ReAct Agent starting: {goal}")
        print("=" * 60)

        while True:
            result = self.step()

            if result in ("done", "max_steps"):
                break

        # If we didn't get an answer but have cycles, summarize
        if not self.state.is_complete and self.state.cycles:
            self.state.final_answer = (
                f"[Reached max steps after {self.state.current_step} attempts. "
                "The agent did not produce a final answer.]"
            )

        # Print summary
        summary = (
            f"\n{'=' * 60}\n"
            f"ReAct Agent finished in {self.state.current_step} steps "
            f"({self.state.elapsed_time():.1f}s)\n"
            f"Cycles completed: {len(self.state.cycles)}\n"
            f"Final answer: {self.state.final_answer}\n"
            f"{'=' * 60}"
        )
        safe_summary = summary.encode('ascii', errors='replace').decode('ascii')
        print(safe_summary)

        return {
            "answer": self.state.final_answer,
            "cycles": self.state.cycles,
            "is_complete": self.state.is_complete,
            "steps": self.state.current_step,
            "time": self.state.elapsed_time(),
        }


# ====================================================================
# Part 4: Quick Test
# ====================================================================

if __name__ == "__main__":
    # Simple test: calculate and write a file
    agent = ReActAgent(max_steps=8)

    result = agent.run(
        goal="Calculate 987 * 654 using the python tool, then write the result to react_test.txt"
    )

    print(f"\n--- Summary ---")
    print(f"Complete: {result['is_complete']}")
    print(f"Answer: {result['answer']}")
    print(f"Cycles: {len(result['cycles'])}")
    for i, cycle in enumerate(result['cycles']):
        print(f"  Cycle {i+1}: {cycle.thought[:60]}...")
