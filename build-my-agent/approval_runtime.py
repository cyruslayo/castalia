"""Human-in-the-loop primitives for the integrated runtime.

Notebook 25 introduces four ideas:
  - approval gates before risky actions
  - human feedback records for iterative revision
  - escalation when the system is uncertain
  - interactive visibility into pending decisions

This module keeps those concepts lightweight and runtime-friendly so the rest of
Castalia can import them without pulling in notebook-only code.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from runtime_contracts import AgentResult, new_id


@dataclass
class ApprovalRequest:
    action: str
    tool_name: str
    params: Dict[str, Any]
    preview: str
    risk: str = "medium"
    actor: str = "agent"
    metadata: Dict[str, Any] = field(default_factory=dict)
    approval_id: str = field(default_factory=lambda: new_id("approval"))
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None
    decision_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeedbackRecord:
    task: str
    draft: str
    feedback: str
    actor: str = "human"
    accepted: bool = True
    feedback_id: str = field(default_factory=lambda: new_id("feedback"))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EscalationRecord:
    reason: str
    task: str
    confidence: float
    actor: str = "agent"
    details: Dict[str, Any] = field(default_factory=dict)
    escalation_id: str = field(default_factory=lambda: new_id("escalation"))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class ApprovalStore:
    """Persist pending approvals and decisions.

    Modes:
      - auto_allow: every request is approved immediately
      - auto_deny: every request is denied immediately
      - manual: requests remain pending until decide() is called
    """

    def __init__(self, mode: str = "auto_deny"):
        assert mode in {"auto_deny", "auto_allow", "manual"}
        self.mode = mode
        self.pending: Dict[str, ApprovalRequest] = {}
        self.decisions: Dict[str, dict] = {}
        self.history: List[dict] = []

    def request(
        self,
        action: str,
        tool_name: str,
        params: dict,
        preview: str,
        risk: str = "medium",
        actor: str = "agent",
        metadata: Optional[dict] = None,
    ) -> ApprovalRequest:
        req = ApprovalRequest(
            action=action,
            tool_name=tool_name,
            params=params,
            preview=preview,
            risk=risk,
            actor=actor,
            metadata=metadata or {},
        )
        if self.mode == "auto_allow":
            req.status = "approved"
            req.decided_at = time.time()
            req.decision_reason = "auto_allow"
            self.decisions[req.approval_id] = {"approved": True, "reason": "auto_allow", "request": req.to_dict()}
        elif self.mode == "auto_deny":
            req.status = "denied"
            req.decided_at = time.time()
            req.decision_reason = "auto_deny"
            self.decisions[req.approval_id] = {"approved": False, "reason": "auto_deny", "request": req.to_dict()}
        else:
            self.pending[req.approval_id] = req
        self.history.append({"event": "requested", "request": req.to_dict()})
        return req

    def decide(self, approval_id: str, approved: bool, reason: str = "") -> dict:
        req = self.pending.pop(approval_id, None)
        if req is None:
            return {"success": False, "error": f"Unknown pending approval: {approval_id}"}
        req.status = "approved" if approved else "denied"
        req.decided_at = time.time()
        req.decision_reason = reason
        decision = {"approved": approved, "reason": reason, "request": req.to_dict()}
        self.decisions[approval_id] = decision
        self.history.append({"event": "decided", **decision})
        return {"success": True, **decision}

    def get(self, approval_id: str) -> Optional[ApprovalRequest]:
        return self.pending.get(approval_id)

    def get_request_dict(self, approval_id: str) -> Optional[dict]:
        pending = self.pending.get(approval_id)
        if pending is not None:
            return pending.to_dict()
        decision = self.decisions.get(approval_id)
        if decision is not None:
            return decision.get("request")
        return None

    def is_approved(self, approval_id: str) -> bool:
        return bool(self.decisions.get(approval_id, {}).get("approved"))

    def pending_items(self) -> List[dict]:
        return [req.to_dict() for req in self.pending.values()]

    def stats(self) -> dict:
        approved = sum(1 for d in self.decisions.values() if d.get("approved"))
        denied = sum(1 for d in self.decisions.values() if not d.get("approved"))
        total = approved + denied + len(self.pending)
        return {
            "mode": self.mode,
            "total_requests": total,
            "pending": len(self.pending),
            "approved": approved,
            "denied": denied,
            "approval_rate": approved / max(1, approved + denied),
        }


class HumanInTheLoopController:
    """Notebook-25 style runtime facade.

    Keeps approval, feedback, and escalation state in one place so AgentRuntime,
    ToolRuntime, or future UI code can share the same human-oversight records.
    """

    def __init__(self, approval_store: Optional[ApprovalStore] = None):
        self.approval_store = approval_store or ApprovalStore(mode="auto_deny")
        self.feedback_log: List[FeedbackRecord] = []
        self.escalations: List[EscalationRecord] = []

    @staticmethod
    def build_preview(tool_name: str, params: Dict[str, Any], max_len: int = 180) -> str:
        rendered = f"{tool_name}({params})"
        return rendered if len(rendered) <= max_len else rendered[: max_len - 3] + "..."

    @staticmethod
    def summarize_tool_result(tool_name: str, tool_result: Any) -> str:
        if getattr(tool_result, "success", False):
            payload = getattr(tool_result, "result", None)
            return f"Approved tool '{tool_name}' executed successfully. Result: {payload}"
        return f"Approved tool '{tool_name}' executed but failed: {getattr(tool_result, 'error', 'unknown error')}"

    def request_tool_approval(
        self,
        tool_name: str,
        params: Dict[str, Any],
        actor: str = "agent",
        risk: str = "medium",
        action: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ApprovalRequest:
        return self.approval_store.request(
            action=action or f"Execute tool '{tool_name}'",
            tool_name=tool_name,
            params=params,
            preview=self.build_preview(tool_name, params),
            risk=risk,
            actor=actor,
            metadata=metadata,
        )

    def record_feedback(self, task: str, draft: str, feedback: str,
                        actor: str = "human", accepted: bool = True) -> FeedbackRecord:
        item = FeedbackRecord(task=task, draft=draft, feedback=feedback, actor=actor, accepted=accepted)
        self.feedback_log.append(item)
        return item

    def revise_with_feedback(
        self,
        task: str,
        draft: str,
        feedback: str,
        llm_fn: Optional[Callable[[List[dict]], str]] = None,
    ) -> dict:
        record = self.record_feedback(task, draft, feedback)

        def _default_llm_fn(messages: List[dict]) -> str:
            from config import get_client, get_model

            response = get_client().chat.completions.create(
                model=get_model(),
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                timeout=180,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            msg = response.choices[0].message
            return (msg.content or getattr(msg, "reasoning", "") or "").strip()

        generator = llm_fn or _default_llm_fn
        messages = [
            {
                "role": "system",
                "content": (
                    "You are revising a draft based on human feedback. "
                    "Return only the revised draft. Preserve factual content unless the feedback asks to change it."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\n"
                    f"Current draft:\n{draft}\n\n"
                    f"Human feedback:\n{feedback}\n\n"
                    "Revised draft:"
                ),
            },
        ]
        revised = generator(messages)
        return {"feedback": record, "revised_draft": revised}

    def maybe_escalate(
        self,
        task: str,
        result: Optional[AgentResult] = None,
        confidence: Optional[float] = None,
        reason: Optional[str] = None,
        actor: str = "agent",
        details: Optional[dict] = None,
    ) -> Optional[EscalationRecord]:
        if reason is None and result is not None:
            if result.metadata.get("requires_human_approval"):
                return None
            if result.metadata.get("approval_denied"):
                return None
            if result.strategy_used in {"blocked", "rate_limited"}:
                return None

        inferred_confidence = confidence
        if inferred_confidence is None:
            if result is None:
                inferred_confidence = 0.5
            elif not result.success:
                inferred_confidence = 0.2
            elif result.errors:
                inferred_confidence = 0.45
            else:
                inferred_confidence = float(result.metadata.get("confidence", 0.85))

        should_escalate = inferred_confidence < 0.5 or bool(reason)
        if not should_escalate:
            return None

        record = EscalationRecord(
            reason=reason or "low_confidence",
            task=task,
            confidence=float(inferred_confidence),
            actor=actor,
            details=details or {},
        )
        self.escalations.append(record)
        return record

    def stats(self) -> dict:
        return {
            "approvals": self.approval_store.stats(),
            "feedback_items": len(self.feedback_log),
            "escalations": len(self.escalations),
        }
