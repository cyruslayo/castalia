"""
Advanced Tool Agent — ReAct-style agent wired to the full ToolRegistry.

This demonstrates Section E of Notebook 13: Testing & Agent Integration.

Architecture:
    User Query → Agent (LLM) → Tool Call → ToolRegistry → Result → Agent → Answer

The agent receives tool descriptions from registry.discover() in its system prompt,
parses JSON tool calls, executes them via the registry, and feeds results back
until it has a final answer.
"""

import json
import re
import time
from typing import List, Dict, Any, Optional

from config import client, get_model
from tool_registry import ToolRegistry, build_production_registry
from tool_definitions import ToolDefinition

# ============================================================================
# AdvancedToolAgent
# ============================================================================

class AdvancedToolAgent:
    """Agent that uses the full tool registry with ReAct-style reasoning.

    The agent loop:
      1. Build system prompt with tool descriptions from registry
      2. Send query to LLM
      3. Parse JSON response for tool call or answer
      4. If tool call: execute via registry, feed result back
      5. If answer: return final response
      6. Repeat until max_steps or answer found
    """

    def __init__(self, registry: ToolRegistry, max_steps: int = 8):
        self.registry = registry
        self.max_steps = max_steps

    def _build_system_prompt(self) -> str:
        """Build system prompt with full tool registry descriptions."""
        tool_desc = self.registry.discover()
        return f"""You are a helpful assistant with access to tools. Use tools when needed.

{tool_desc}

RESPONSE FORMAT:
When you need a tool, respond with EXACTLY this JSON (no extra text):
{{"thought": "why I need this tool", "tool": "tool_name", "args": {{"param": "value"}}}}

When you have the final answer, respond with:
{{"thought": "reasoning", "answer": "your final answer to the user"}}

Rules:
- Use tools for computation, data manipulation, encoding — don't guess
- If a tool returns an error, read the error message carefully and retry with corrected inputs
- You may chain multiple tool calls to solve complex problems
- Always respond with valid JSON only"""

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response with multiple fallback strategies."""
        # Strategy 1: direct parse
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Strategy 2: find JSON block in markdown
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: find first JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def run(self, query: str, verbose: bool = True) -> dict:
        """Run a query through the agent loop.

        Returns:
            dict with 'answer', 'steps', 'tool_calls', 'total_time'
        """
        start_time = time.time()
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query},
        ]
        steps = []
        tool_calls = 0

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"QUERY: {query}")
            print(f"{'=' * 60}")

        for step_num in range(self.max_steps):
            # Call LLM
            try:
                response = client.chat.completions.create(
                    model=get_model(),
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
                    extra_body={'chat_template_kwargs': {'enable_thinking': False}},
                )
                msg = response.choices[0].message
                response_text = msg.content or ""
            except Exception as e:
                if verbose:
                    print(f"\n  LLM Error: {e}")
                steps.append({
                    "step": step_num,
                    "type": "error",
                    "content": f"LLM error: {e}",
                })
                return {
                    "answer": f"Failed: LLM error — {e}",
                    "steps": steps,
                    "tool_calls": tool_calls,
                    "total_time": time.time() - start_time,
                }

            # Parse JSON response
            parsed = self._extract_json(response_text)

            if parsed is None:
                # LLM didn't produce JSON — treat as direct answer
                if verbose:
                    print(f"\n  Step {step_num}: [Direct Answer]")
                    print(f"    {response_text[:200]}...")
                steps.append({
                    "step": step_num,
                    "type": "answer",
                    "content": response_text,
                })
                return {
                    "answer": response_text,
                    "steps": steps,
                    "tool_calls": tool_calls,
                    "total_time": time.time() - start_time,
                }

            # Check for answer
            if "answer" in parsed:
                if verbose:
                    print(f"\n  Step {step_num}: [Answer]")
                    print(f"    {parsed['answer']}")
                steps.append({
                    "step": step_num,
                    "type": "answer",
                    "content": parsed["answer"],
                })
                return {
                    "answer": parsed["answer"],
                    "steps": steps,
                    "tool_calls": tool_calls,
                    "total_time": time.time() - start_time,
                }

            # Check for tool call
            if "tool" in parsed:
                tool_name = parsed["tool"]
                tool_args = parsed.get("args", {})
                thought = parsed.get("thought", "")

                if verbose:
                    print(f"\n  Step {step_num}: [Tool Call: {tool_name}]")
                    print(f"    Thought: {thought}")
                    print(f"    Args: {tool_args}")

                # Execute tool via registry
                tool_result = self.registry.call(tool_name, **tool_args)
                tool_calls += 1

                if verbose:
                    status = "OK" if tool_result.success else "ERROR"
                    print(f"    Result: [{status}] {tool_result.to_dict()}")

                steps.append({
                    "step": step_num,
                    "type": "tool",
                    "thought": thought,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result.to_dict(),
                })

                # Feed result back to agent
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {json.dumps(tool_result.to_dict())}",
                })
            else:
                # No recognizable action
                if verbose:
                    print(f"\n  Step {step_num}: [Unknown Response]")
                steps.append({
                    "step": step_num,
                    "type": "unknown",
                    "content": response_text,
                })
                return {
                    "answer": response_text,
                    "steps": steps,
                    "tool_calls": tool_calls,
                    "total_time": time.time() - start_time,
                }

        # Max steps reached
        if verbose:
            print(f"\n  [Max Steps Reached: {self.max_steps}]")
        return {
            "answer": f"Max steps ({self.max_steps}) reached without final answer.",
            "steps": steps,
            "tool_calls": tool_calls,
            "total_time": time.time() - start_time,
        }


# ============================================================================
# Demo: Run the agent against test queries
# ============================================================================

def run_demos():
    """Run demo queries through the AdvancedToolAgent."""
    registry = build_production_registry()
    agent = AdvancedToolAgent(registry, max_steps=6)

    # Query 1: Simple math
    print("\n" + "=" * 60)
    print("DEMO 1: Calculator via Tool Use")
    print("=" * 60)
    result1 = agent.run("What is 2 raised to the power of 16? Use a tool to compute it.")
    print(f"\nFinal answer: {result1['answer']}")
    print(f"Tool calls: {result1['tool_calls']}")
    print(f"Time: {result1['total_time']:.1f}s")

    # Query 2: File I/O
    print("\n" + "=" * 60)
    print("DEMO 2: File Write and Read")
    print("=" * 60)
    result2 = agent.run(
        "Write the text 'Hello from the agent!' to a file called 'agent_greeting.txt', "
        "then read it back and tell me what it says."
    )
    print(f"\nFinal answer: {result2['answer']}")
    print(f"Tool calls: {result2['tool_calls']}")
    print(f"Time: {result2['total_time']:.1f}s")

    # Query 3: Knowledge base search
    print("\n" + "=" * 60)
    print("DEMO 3: Knowledge Base Search")
    print("=" * 60)
    result3 = agent.run("Search the knowledge base for facts about Paris.")
    print(f"\nFinal answer: {result3['answer']}")
    print(f"Tool calls: {result3['tool_calls']}")
    print(f"Time: {result3['total_time']:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("REGISTRY STATISTICS AFTER ALL DEMOS")
    print("=" * 60)
    stats = registry.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_demos()
