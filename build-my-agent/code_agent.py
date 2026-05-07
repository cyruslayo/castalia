"""
Production CodeAgent — an agent that writes, runs, and iterates on code.

Uses ReAct-style reasoning to:
  1. Analyze the user's problem
  2. Write Python code to solve it
  3. Execute via CodeExecutor (full security pipeline)
  4. Inspect output / error
  5. If error, analyze and write fixed code
  6. Repeat until successful or max cycles

Integrates with the existing Castalia agent architecture:
  - Uses the same LLM client from config.py
  - Compatible with the existing agent_loop.py patterns
  - Produces structured CodeExecutionResult objects
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from config import get_model
from code_executor import CodeExecutor, CodeExecutionResult


# ─── Data Classes ────────────────────────────────────────────────

@dataclass
class CodeCycle:
    """One iteration of the write-run-fix loop."""
    cycle_number: int
    thought: str
    code: str
    result: Optional[CodeExecutionResult] = None
    is_final: bool = False

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle_number,
            "thought": self.thought,
            "code": self.code[:200] + "..." if len(self.code) > 200 else self.code,
            "success": self.result.execution_succeeded if self.result else None,
            "exit_code": self.result.exit_code if self.result else None,
            "execution_time_ms": self.result.execution_time_ms if self.result else None,
            "is_final": self.is_final,
        }


@dataclass
class CodeAgentState:
    """Full state of a CodeAgent session."""
    goal: str
    cycles: list = field(default_factory=list)
    current_cycle: int = 0
    max_cycles: int = 5
    is_complete: bool = False
    final_answer: str = ""
    total_time_ms: float = 0.0

    @property
    def last_cycle(self) -> Optional[CodeCycle]:
        return self.cycles[-1] if self.cycles else None

    @property
    def success_cycles(self) -> int:
        return sum(1 for c in self.cycles if c.result and c.result.execution_succeeded)

    def summary(self) -> dict:
        return {
            "goal": self.goal,
            "total_cycles": len(self.cycles),
            "successful_cycles": self.success_cycles,
            "is_complete": self.is_complete,
            "final_answer": self.final_answer[:200] if self.final_answer else "",
            "total_time_ms": round(self.total_time_ms, 2),
        }


# ─── System Prompt ───────────────────────────────────────────────

CODE_AGENT_SYSTEM_PROMPT = """\
You are a CodeAgent — an expert Python programmer that writes code to solve computational problems.

Your process:
1. Think through the problem carefully
2. Write clean, correct Python code
3. The code will be executed in a sandbox and you'll see the output
4. If there's an error, analyze it and write fixed code
5. Repeat until the code produces the correct result

RULES:
- You may ONLY use these modules: math, json, statistics, random, itertools, functools, operator, collections, datetime, time, calendar, string, re, copy, pprint, enum, typing, dataclasses, hashlib, hmac, secrets, bisect, heapq, array, struct, codecs, decimal, fractions, cmath, textwrap, unicodedata
- You may NOT use: os, sys, subprocess, socket, requests, urllib, http, pickle, ctypes, multiprocessing, threading, importlib, pathlib, shutil
- Your code MUST be complete and self-contained — include all imports, data, and logic
- Your code MUST print the final answer using print()
- Your code MUST NOT attempt to read files, access the network, or interact with the system
- Keep code concise and well-commented
- Handle edge cases (empty inputs, zero division, etc.)

RESPONSE FORMAT:
You MUST respond with valid JSON containing exactly these fields:
{
    "thought": "Your reasoning about the problem and approach",
    "code": "The Python code to execute (use \\n for newlines)",
    "done": true/false
}

If "done" is true, you believe the code solves the problem and no more cycles are needed.
If "done" is false, you expect the code might need refinement after seeing the output.
"""

CODE_FIX_PROMPT_TEMPLATE = """\
Your previous code produced an error. Analyze the error and write fixed code.

ORIGINAL CODE:
{original_code}

ERROR OUTPUT:
{error_output}

Think through what went wrong, then write corrected code.
The corrected code should handle the error case properly.

RESPONSE FORMAT (JSON):
{{
    "thought": "Analysis of the error and your fix approach",
    "code": "The corrected Python code",
    "done": true/false
}}
"""


# ─── CodeAgent ───────────────────────────────────────────────────

class CodeAgent:
    """
    Agent that writes and iteratively refines code to solve problems.

    Usage:
        agent = CodeAgent(executor=my_executor)
        result = agent.run("Calculate the 50th Fibonacci number")
        print(result.final_answer)
    """

    def __init__(
        self,
        executor: Optional[CodeExecutor] = None,
        max_cycles: int = 5,
        model: Optional[str] = None,
        temperature: float = 0.3,  # Lower temp for code generation
        max_tokens: int = 2048,
    ):
        self.executor = executor or CodeExecutor(timeout=30)
        self.max_cycles = max_cycles
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = get_model()

    def run(self, goal: str, user_id: str = "code-agent") -> CodeAgentState:
        """
        Run the full code generation loop.

        Args:
            goal: Natural language description of the problem to solve
            user_id: Identifier for audit logging

        Returns:
            CodeAgentState with full execution trace
        """
        start_time = time.monotonic()
        state = CodeAgentState(goal=goal, max_cycles=self.max_cycles)
        last_code = ""
        last_error = ""

        for cycle_num in range(1, self.max_cycles + 1):
            state.current_cycle = cycle_num

            # ── Generate code ─────────────────────────────────────
            if cycle_num == 1:
                prompt = CODE_AGENT_SYSTEM_PROMPT + f"\n\nPROBLEM TO SOLVE: {goal}"
            else:
                prompt = (CODE_AGENT_SYSTEM_PROMPT + "\n\n" +
                         CODE_FIX_PROMPT_TEMPLATE.format(
                             original_code=last_code,
                             error_output=last_error
                         ) + f"\n\nPROBLEM TO SOLVE: {goal}")

            llm_response = self._call_llm(prompt)
            parsed = self._parse_response(llm_response, goal)

            cycle = CodeCycle(
                cycle_number=cycle_num,
                thought=parsed.get("thought", ""),
                code=parsed.get("code", ""),
            )
            state.cycles.append(cycle)
            last_code = cycle.code

            # ── Execute code ──────────────────────────────────────
            if not last_code.strip():
                cycle.result = CodeExecutionResult(
                    success=False, output="",
                    error="Agent produced empty code",
                    exit_code=-1, execution_time_ms=0,
                )
                last_error = "Empty code generated"
                continue

            exec_result = self.executor.execute(last_code, user_id=user_id)
            cycle.result = exec_result

            if exec_result.execution_succeeded:
                # Success! Extract the answer
                cycle.is_final = True
                state.is_complete = True
                state.final_answer = self._extract_answer(exec_result.output, goal)
                state.total_time_ms = (time.monotonic() - start_time) * 1000
                return state

            # Failed — prepare for retry
            last_error = exec_result.error
            state.total_time_ms = (time.monotonic() - start_time) * 1000

        # Max cycles reached without success
        state.is_complete = True
        state.total_time_ms = (time.monotonic() - start_time) * 1000
        state.final_answer = (
            f"Failed after {self.max_cycles} cycles. "
            f"Last error: {last_error[:500]}"
        )
        return state

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the code generation prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # Hermes model quirk: check reasoning field
            content = response.choices[0].message.content
            if not content:
                content = getattr(response.choices[0].message, 'reasoning', '')
            return content or ""
        except Exception as e:
            return json.dumps({
                "thought": f"LLM error: {str(e)}",
                "code": "",
                "done": True,
            })

    def _parse_response(self, response: str, goal: str) -> dict:
        """Parse LLM response into structured code + thought."""
        # Try direct JSON parse
        try:
            data = json.loads(response)
            if "code" in data and "thought" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        import re
        match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try extracting JSON from braces
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                pass

        # Failsafe: wrap the whole response as a thought with no code
        return {
            "thought": response[:500],
            "code": "",
            "done": True,
        }

    def _extract_answer(self, output: str, goal: str) -> str:
        """Extract the final answer from execution output."""
        # Remove the output header from format_for_llm
        lines = output.strip().split('\n')
        # Skip any header lines (bracket-prefixed from format_for_llm)
        content_lines = [l for l in lines if not l.startswith('[') and l.strip()]
        return '\n'.join(content_lines[-10:])  # Last 10 lines as answer


# ─── Comparison Runner ───────────────────────────────────────────

def compare_code_vs_direct(goal: str, code_agent: CodeAgent, max_cycles: int = 5) -> dict:
    """
    Compare CodeAgent output vs direct LLM reasoning on the same task.

    Returns dict with both approaches and a verdict.
    """
    start = time.monotonic()

    # Code agent approach
    code_result = code_agent.run(goal)
    code_time = time.monotonic() - start

    start = time.monotonic()

    # Direct reasoning approach
    client = code_agent.client
    prompt = f"Solve this problem using only your reasoning (no code execution): {goal}"
    try:
        response = client.chat.completions.create(
            model=code_agent.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        if not content:
            content = getattr(response.choices[0].message, 'reasoning', 'No response')
        direct_answer = content or "No response"
    except Exception as e:
        direct_answer = f"LLM error: {str(e)}"
    direct_time = time.monotonic() - start

    return {
        "goal": goal,
        "code_agent": {
            "answer": code_result.final_answer[:500],
            "cycles": len(code_result.cycles),
            "success": code_result.is_complete and code_result.success_cycles > 0,
            "time_seconds": round(code_time, 2),
        },
        "direct_reasoning": {
            "answer": direct_answer[:500],
            "time_seconds": round(direct_time, 2),
        },
    }


# ─── Self-test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=== CodeAgent Self-Test ===\n")

    executor = CodeExecutor(timeout=10, rate_limit=20)
    agent = CodeAgent(executor=executor, max_cycles=3)

    # Test 1: Fibonacci
    print("Test 1: 50th Fibonacci number")
    state = agent.run("Calculate the 50th Fibonacci number using an iterative approach")
    print(f"  Cycles: {len(state.cycles)}")
    print(f"  Success: {state.is_complete and state.success_cycles > 0}")
    print(f"  Answer: {state.final_answer[:100]}")
    print(f"  Time: {state.total_time_ms:.0f}ms")
    for c in state.cycles:
        print(f"  Cycle {c.cycle_number}: thought='{c.thought[:60]}...', code_len={len(c.code)}, success={c.result.execution_succeeded if c.result else 'N/A'}")
    print()

    # Test 2: Statistics
    print("Test 2: Standard deviation")
    state = agent.run("Calculate the standard deviation of [12, 15, 18, 21, 24, 27, 30, 33, 36, 39]")
    print(f"  Cycles: {len(state.cycles)}")
    print(f"  Success: {state.is_complete and state.success_cycles > 0}")
    print(f"  Answer: {state.final_answer[:100]}")
    print(f"  Time: {state.total_time_ms:.0f}ms")
    print()

    # Test 3: Comparison
    print("Test 3: Code vs Direct reasoning")
    comparison = compare_code_vs_direct(
        "What is 2 raised to the power of 47?",
        agent
    )
    print(f"  Code agent: {comparison['code_agent']['success']} "
          f"({comparison['code_agent']['cycles']} cycles, "
          f"{comparison['code_agent']['time_seconds']}s)")
    print(f"  Direct: {comparison['direct_reasoning']['answer'][:80]}... "
          f"({comparison['direct_reasoning']['time_seconds']}s)")
    print()

    print("CodeAgent tests complete!")
