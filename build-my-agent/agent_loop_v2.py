"""AgentLoopV2 — registry-backed successor to the original AgentLoop.

This keeps the educational think/answer/use_tool loop shape, but replaces the
legacy tools.TOOLS path with ToolRuntime/ToolRegistry so all wired tools are
available from the same loop.
"""

from __future__ import annotations

import time
from typing import Optional

from config import get_client, get_model
from memory import MemoryManager
from parser import parse_response
from registry_bootstrap import build_full_tool_registry
from runtime_contracts import AgentResult, StepRecord
from tool_runtime import ToolRuntime


BASE_PROMPT = """You are a reasoning agent. Solve problems step by step.

On each turn, respond with EXACTLY one JSON object.

Think:
{"action": "think", "thought": "your reasoning"}

Use a tool:
{"action": "use_tool", "tool": "tool_name", "params": {}}

Answer:
{"action": "answer", "answer": "final answer"}

Rules:
- Use tools for computation, data, files, code execution, and search.
- After a tool result, reason about it or answer.
- Output JSON only.
"""


class AgentLoopV2:
    def __init__(self, task: str, tool_runtime: Optional[ToolRuntime] = None,
                 max_steps: int = 10, memory_strategy: str = "sliding", **memory_kwargs):
        self.task = task
        self.max_steps = max_steps
        self.tool_runtime = tool_runtime or ToolRuntime(build_full_tool_registry())
        self.memory = MemoryManager(strategy=memory_strategy, **memory_kwargs)
        system = BASE_PROMPT + "\n\n" + self.tool_runtime.discover()
        self.memory.add_dict("system", system)
        self.memory.add_dict("user", f"Task: {task}")
        self.steps = []
        self.tool_calls_start = len(self.tool_runtime.call_records)

    def _call_llm(self, messages: list[dict]) -> str:
        client = get_client()
        response = client.chat.completions.create(
            model=get_model(),
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    def run(self) -> AgentResult:
        started = time.time()
        answer = ""
        success = False
        errors = []

        for i in range(self.max_steps):
            raw = self._call_llm(self.memory.get_context_dicts())
            parsed = parse_response(raw)
            action = parsed.get("action", "think")
            self.steps.append(StepRecord(step=i, action=action, raw_response=raw, content=parsed).to_dict())

            if action == "answer":
                answer = parsed.get("answer", "")
                success = True
                self.memory.add_dict("assistant", raw)
                break

            if action == "use_tool":
                tool = parsed.get("tool", "")
                params = parsed.get("params", {})
                result = self.tool_runtime.call(tool, params)
                self.memory.add_dict("assistant", raw)
                self.memory.add_dict("user", f"Tool result: {result.to_dict()}\nContinue. If complete, answer.")
                continue

            thought = parsed.get("thought", raw)
            self.memory.add_dict("assistant", raw)
            self.memory.add_dict("user", "Continue reasoning. Use a tool if needed, otherwise answer.")

        if not success:
            answer = answer or "Max steps reached without final answer."
            errors.append(answer)

        return AgentResult(
            answer=answer,
            success=success,
            strategy_used="agent_loop_v2",
            steps=self.steps,
            tool_calls=[r.to_dict() for r in self.tool_runtime.call_records[self.tool_calls_start:]],
            metadata={"duration_seconds": time.time() - started, "max_steps": self.max_steps},
            errors=errors,
            started_at=started,
        ).finish()


if __name__ == "__main__":
    agent = AgentLoopV2("Use calculator to compute 2+3*4 and answer.", max_steps=4)
    print(agent.run().to_dict())
