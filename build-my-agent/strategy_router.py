"""Heuristic strategy router for AgentRuntime.

This is intentionally simple for Milestone A/B. Notebook 27 can replace or
augment it with model routing, cost-aware policies, and cached classifiers.
"""

from __future__ import annotations

import re


class StrategyRouter:
    def __init__(self, default: str = "react"):
        self.default = default

    def select(self, task: str, requested: str = "auto") -> str:
        if requested and requested != "auto":
            return requested

        text = task.lower()

        if any(k in text for k in ["debug", "fix this code", "write code", "unit test", "python function", "implement function"]):
            return "code"

        if any(k in text for k in ["csv", "json data", "dataset", "dataframe", "analyze data", "schema", "columns", "rows"]):
            return "data"

        if any(k in text for k in ["mcp", "model context protocol", "remote tool server", "tool server"]):
            return "mcp"

        if any(k in text for k in ["research", "cite", "sources", "latest", "current", "web", "internet"]):
            return "research"

        if any(k in text for k in ["critique", "revise", "improve quality", "polish", "review"]):
            return "reflection"

        multi_step_markers = ["first", "then", "finally", "step", "plan", "dependencies", "multi-step"]
        if len(text.split()) > 80 or sum(1 for k in multi_step_markers if k in text) >= 2:
            return "plan_execute"

        if re.search(r"\b(compare|tradeoff|pros and cons|multiple approaches)\b", text):
            return "reflection"

        return self.default
