"""Castalia Scholar capstone scaffold (Notebooks 32-37 foundation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from blackboard_runtime import BlackboardRuntime
from runtime_contracts import AgentRequest


CAPSTONE_KEYS = [
    "task", "research_questions", "sources", "retrieved_chunks", "claims",
    "evidence_table", "contradictions", "outline", "draft", "review_feedback",
    "final_report", "citations",
]


@dataclass
class ScholarResult:
    final_report: str
    success: bool
    blackboard: Dict[str, Any] = field(default_factory=dict)
    stages: Dict[str, Any] = field(default_factory=dict)


class CastaliaScholarOrchestrator:
    """Thin orchestrator scaffold over AgentRuntime + BlackboardRuntime."""

    def __init__(self, runtime, blackboard: BlackboardRuntime = None):
        self.runtime = runtime
        self.blackboard = blackboard or BlackboardRuntime(name="castalia_scholar")

    def run(self, task: str) -> ScholarResult:
        self.blackboard.publish("task", task, author="orchestrator")

        research = self.runtime.run(AgentRequest(task=f"Research with citations: {task}", strategy="research"))
        self.blackboard.publish("sources", research.artifacts, author="retriever")

        analysis = self.runtime.run(AgentRequest(task=f"Analyze evidence and contradictions for: {task}\nSources: {research.answer[:3000]}", strategy="reflection"))
        self.blackboard.publish("claims", analysis.answer, author="analysis")

        draft = self.runtime.run(AgentRequest(task=f"Write a clear report for: {task}\nAnalysis: {analysis.answer[:4000]}", strategy="plan_execute"))
        self.blackboard.publish("draft", draft.answer, author="writer")

        review = self.runtime.run(AgentRequest(task=f"Review this report for accuracy, completeness, citations, and clarity:\n{draft.answer[:5000]}", strategy="reflection"))
        self.blackboard.publish("review_feedback", review.answer, author="reviewer")

        # Initial scaffold: final report is the draft plus review note. Notebook 37 can
        # make this an explicit revise-until-pass loop.
        final_report = draft.answer
        self.blackboard.publish("final_report", final_report, author="orchestrator")

        return ScholarResult(
            final_report=final_report,
            success=draft.success,
            blackboard=self.blackboard.snapshot(),
            stages={"research": research.to_dict(), "analysis": analysis.to_dict(), "draft": draft.to_dict(), "review": review.to_dict()},
        )
