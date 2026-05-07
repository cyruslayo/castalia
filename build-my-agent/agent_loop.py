"""
Agent Loop - The main execution engine.

This module implements the core agent loop that:
1. Maintains the agent state (AgentState)
2. Calls the LLM on each step
3. Parses the response to determine the next action
4. Handles tool execution
5. Manages memory with a sliding window
6. Terminates when the goal is complete or max steps reached
"""

import time  # For timing each step
import subprocess  # For running external commands (used in testing)

from typing import List, Optional, Dict, Any
from state import AgentState  # The state dataclass that tracks the entire conversation
from parser import parse_response  # Our 4-strategy fallback parser
from config import get_client, get_model, LLM_CONFIG  # LLM configuration
from memory import MemoryManager, Message  # Unified memory controller
from guard import guarded_execute_tool  # Layer 1 (name) + Layer 2 (params) protection
from tools import generate_tools_instruction  # System prompt generator

# --- LLM Call Function ---
def call_llm(messages: list) -> str:
    """
    Send messages to the LLM and get a response.
    
    Uses the OpenAI client configured for our vLLM endpoint.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        The text content of the assistant's response
    """
    client = get_client()
    model = get_model()
    
    # Send the messages to the LLM
    # We use a high max_tokens to allow the model to think through complex problems
    # We set temperature for a balance of creativity and reliability
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,  # Cap to avoid exceeding context window
        temperature=0.7   # Moderate randomness
    )
    
    # Return just the text content of the response
    return response.choices[0].message.content


# --- System prompt that tells the LLM how to behave ---
SYSTEM_PROMPT = """You are a reasoning agent. Solve problems step by step.

On each turn, respond with EXACTLY one JSON object (no other text, no markdown, no explanation).

Option 1 - Think (reason about the problem):
{"action": "think", "thought": "your reasoning here"}

Option 2 - Answer (provide final answer):
{"action": "answer", "answer": "your final answer here"}

Option 3 - Use a tool (execute an action and get the result back):
{"action": "use_tool", "tool": "tool_name", "params": { ... }}

CRITICAL: The 'tool' key must be at the TOP LEVEL of the JSON object, NOT inside 'params'.

Example for using the python tool:
{"action": "use_tool", "tool": "python", "params": {"code": "123 * 456"}}

Example for using the read_file tool:
{"action": "use_tool", "tool": "read_file", "params": {"filepath": "data.txt"}}

Rules:
- Use "think" to break down the problem, consider approaches, and work through steps.
- Use "answer" ONLY when you are confident you have the complete solution.
- Use "use_tool" when you need external information or computation.
- Each "think" step should make progress toward the answer.
- After a tool returns a result, use "think" to process it, or "answer" if done.
- Respond with ONLY the JSON object, no additional text, no markdown code blocks.
""" + generate_tools_instruction()


class AgentLoop:
    """
    The core agent execution loop.
    
    This is the main engine that drives the agent. It maintains state,
    calls the LLM, parses responses, and handles tool execution.
    
    The agent works in a cycle:
    1. Build a prompt from the current state (with memory management)
    2. Call the LLM with the prompt
    3. Parse the response to get the action
    4. Handle the action (think/answer/use_tool)
    5. Repeat until the goal is complete or max steps reached
    
    The state tracks:
    - The goal the agent is trying to achieve
    - The conversation history (messages)
    - The current step number
    - Whether the agent is complete
    - The final answer
    - Metadata (like start time)
    """
    
    def __init__(self, goal: str, system_prompt: str = SYSTEM_PROMPT,
                 max_steps: int = 10, memory_strategy: str = "sliding",
                 **memory_kwargs):
        """
        Initialize the agent with a goal.
        
        Args:
            goal: The task the agent needs to accomplish
            system_prompt: The system prompt that guides the LLM's behavior
            max_steps: Maximum number of steps before the agent stops (safety limit)
            memory_strategy: One of "full", "sliding", "summarizing", "importance"
            **memory_kwargs: Passed to the MemoryManager (e.g., window_size=10)
        """
        # Create the initial state with the goal
        # The state starts with just the system message and the user's goal
        self.state = AgentState(
            goal=goal,
            messages=[
                {"role": "system", "content": system_prompt},  # How to behave
                {"role": "user", "content": f"Task: {goal}\n\nStart by reasoning about this task. What do you need to do? Show me your work step by step."}  # The actual task
            ],
            max_steps=max_steps,
        )
        # Memory manager replaces the old window_size approach
        self.memory = MemoryManager(strategy=memory_strategy, **memory_kwargs)
        # Seed the memory with the initial messages
        for msg in self.state.messages:
            self.memory.add_dict(msg["role"], msg["content"])
    
    def run(self, max_steps: Optional[int] = None) -> AgentState:
        """
        Run the agent loop until the goal is complete or max steps reached.
        
        This is the main entry point. It runs the agent step by step,
        handling the core loop:
        1. Check if we should stop (complete or max steps)
        2. Take a step (call LLM, parse, handle action)
        3. Repeat
        
        Args:
            max_steps: Optional override for max steps (if not set, uses the state's max_steps)
            
        Returns:
            The final state after the agent completes (or gives up)
        """
        # Allow overriding max_steps for testing
        if max_steps is not None:
            self.state.max_steps = max_steps
        
        print(f"\n{'=' * 60}")
        print(f"Agent starting: {self.state.goal}")
        print(f"{'=' * 60}")
        
        start_time = time.time()  # Record when the agent started
        
        # Main loop: keep going until we should stop
        while not self._should_stop():
            self.step()  # Take one step in the reasoning process
        
        # If we hit max steps without completing, extract the best answer we have
        if not self.state.is_complete:
            # Check if the last step has a response we can use
            last = self.state.steps[-1].get("raw_response", "")
            if not self.state.final_answer:
                self.state.final_answer = f"[Incomplete] Last output: {last[:100]}"
            self.state.is_complete = True
            return self.state
        
        return self.state
    
    def _should_stop(self) -> bool:
        """
        Determine if the agent should stop.
        
        The agent stops when:
        1. It has completed the goal (is_complete is True), OR
        2. It has reached the maximum number of steps (safety limit)
        
        Returns:
            True if the agent should stop, False otherwise
        """
        if self.state.is_complete:
            elapsed = time.time() - LLM_CONFIG.get('start_time', time.time())
            print(f"\n{'=' * 60}")
            print(f"Agent finished in {self.state.current_step} steps ({elapsed:.1f}s)")
            print(f"Answer: {self.state.final_answer[:200]}")  # Truncate long answers
            mem = self.get_memory_stats()
            ctx_msgs = len(self.memory.get_context())
            print(f"Memory: strategy={mem['strategy']}, context={ctx_msgs} msgs, tokens={mem.get('estimated_tokens', 'N/A')}")
            print(f"{'=' * 60}")
            return True
        
        if self.state.current_step >= self.state.max_steps:
            elapsed = time.time() - LLM_CONFIG.get('start_time', time.time())
            print(f"\n{'=' * 60}")
            print(f"Agent stopped after {self.state.max_steps} steps ({elapsed:.1f}s)")
            mem = self.get_memory_stats()
            ctx_msgs = len(self.memory.get_context())
            print(f"Memory: strategy={mem['strategy']}, context={ctx_msgs} msgs, tokens={mem.get('estimated_tokens', 'N/A')}")
            print(f"{'=' * 60}")
            return True
        
        return False
    
    def step(self):
        """
        Take one step in the reasoning process.
        
        This is the core of the agent loop. Each step:
        1. Calls the LLM with the current state (with memory management)
        2. Parses the response to get the action (think/answer/use_tool)
        3. Handles the action:
           - think: Add the thought to the conversation and continue
           - answer: Set the final answer and mark as complete
           - use_tool: Execute the tool and add the result to the conversation
        4. Records the step in the state history
        
        The step method is the "brain" of the agent. It's where the magic happens.
        """
        step_start = time.time()  # Time this step for performance tracking
        
        # Call the LLM with a memory-managed prompt (avoids context overflow)
        trimmed_messages = self.memory.get_context_dicts()
        response_text = call_llm(trimmed_messages)
        
        step_time = time.time() - step_start  # How long this step took
        print(f"  Step {self.state.current_step + 1} ({step_time:.1f}s): {response_text[:100]}...")  # Show a snippet of the response
        
        # Parse the response to get the action
        # The parser uses 4 fallback strategies to handle various LLM output formats
        parsed = parse_response(response_text)
        action = parsed.get("action", "think")  # Default to "think" if parsing fails
        
        # Record this step in the history
        step_record = {
            "step": self.state.current_step,
            "action": action,
            "raw_response": response_text,  # Keep the full response for debugging
        }
        
        # Act on the decision
        if action == "think":
            thought = parsed.get("thought", "")
            # Show what the agent was thinking (truncated for readability)
            display = thought.replace('\\n', ' ') if len(thought) > 120 else thought
            print(f"  Step {self.state.current_step + 1} [THINK]: {display[:120]}")
            
            # Keep the conversation going by adding the thought to the messages
            self.state.messages.append({"role": "assistant", "content": response_text})
            self.state.messages.append({
                "role": "user",
                "content": "Continue reasoning. When you have the answer, use the answer action."
            })
            self.memory.add_dict("assistant", response_text)
            self.memory.add_dict("user", "Continue reasoning. When you have the answer, use the answer action.")
            
        elif action == "answer":
            answer = parsed.get("answer", "")
            print(f"  Step {self.state.current_step + 1} [ANSWER]: {answer[:200]}")
            
            # Mark as complete and set the final answer
            self.state.is_complete = True
            self.state.final_answer = answer
            
            # The conversation is done, no need to add more messages
            self.state.messages.append({"role": "assistant", "content": response_text})
            self.memory.add_dict("assistant", response_text)
            
        elif action == "use_tool":
            tool_name = parsed.get("tool", "")
            params = parsed.get("params", {})
            
            # Show what tool is being used
            print(f"  Step {self.state.current_step + 1} [TOOL: {tool_name}]")
            
            # Execute the tool and get the result
            result = guarded_execute_tool(tool_name, params)
            print(f"  [TOOL RESULT]: {str(result)[:200]}")  # Show the result (truncated)
            
            # Add the tool result to the conversation
            self.state.messages.append({"role": "assistant", "content": response_text})
            self.state.messages.append({
                "role": "user",
                "content": f"Tool '{tool_name}' returned: {str(result)[:500]}\n\nNow use this result to continue your reasoning. If you have the final answer, use the answer action."
            })
            self.memory.add_dict("assistant", response_text)
            self.memory.add_dict("user", f"Tool '{tool_name}' returned: {str(result)[:500]}\n\nNow use this result to continue your reasoning. If you have the final answer, use the answer action.")
        
        # Update the state: increment step count and record this step
        self.state.current_step += 1
        self.state.steps.append(step_record)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Return diagnostic info about current memory usage.
        
        Useful for monitoring token consumption during long runs.
        """
        return self.memory.stats()

    def switch_memory_strategy(self, new_strategy: str, **kwargs) -> Dict[str, Any]:
        """
        Swap memory strategy mid-run without losing history.
        
        Example: start with 'sliding' for speed, switch to 'summarizing'
        once the conversation gets long.
        """
        return self.memory.switch_strategy(new_strategy, **kwargs)

    def get_state(self) -> AgentState:
        """
        Get the current state of the agent.
        
        Returns:
            The current AgentState object
        """
        return self.state


# --- Test it (with the real LLM) ---
if __name__ == "__main__":
    # Create an agent with a goal, using sliding window memory (window=10)
    agent = AgentLoop(
        goal="Calculate what 987 * 654 equals using the python tool, then write the result to 'calculation.txt' using write_file, and finally read it back with read_file to confirm.",
        memory_strategy="sliding",
        window_size=10,
    )
    
    # Run the agent
    state = agent.run()
    
    # Print the final state
    print(f"\nFinal result:")
    print(state)
