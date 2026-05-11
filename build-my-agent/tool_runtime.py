"""Canonical tool runtime for the integrated agent system."""

from __future__ import annotations

from typing import Any, Dict, Optional

from approval_runtime import ApprovalStore, HumanInTheLoopController
from runtime_contracts import ToolCallRecord
from tool_registry import ToolRegistry, ToolResult


class ToolRuntime:
    """Validated, observable dispatch wrapper around ToolRegistry.

    ToolRegistry remains responsible for schemas and function execution.  This
    runtime adds a stable interface for agents, optional safety policy checks,
    event emission, canonical call records, and Notebook-25 approval gating.
    """

    def __init__(self, registry: ToolRegistry, safety=None, event_bus=None,
                 approval_store: Optional[ApprovalStore] = None):
        self.registry = registry
        self.safety = safety
        self.event_bus = event_bus
        self.approval_store = approval_store
        self.hitl = HumanInTheLoopController(approval_store) if approval_store is not None else None
        self.call_records: list[ToolCallRecord] = []
        self.request_context: Dict[str, Any] = {}

    def discover(self, category: Optional[str] = None) -> str:
        return self.registry.discover(category=category)

    def bind_request_context(self, **context) -> None:
        self.request_context = {k: v for k, v in context.items() if v is not None}

    def clear_request_context(self) -> None:
        self.request_context = {}

    def _request_id(self) -> Optional[str]:
        return self.request_context.get("request_id")

    def call(self, tool_name: str, params: Optional[Dict[str, Any]] = None,
             actor: str = "agent", approved: bool = False) -> ToolResult:
        params = params or {}

        if self.event_bus:
            self.event_bus.emit(
                "tool.call.requested",
                actor,
                {"tool": tool_name, "params": params},
                request_id=self._request_id(),
            )

        # Notebook 25: if a tool needs approval and a store is configured, create
        # a pending approval instead of returning a generic validation failure.
        if self.safety is not None and getattr(self.safety.tool_validator, "policies", None):
            policy = self.safety.tool_validator.policies.get(tool_name)
            if policy and policy.requires_approval and not approved and self.hitl is not None:
                approval_metadata = dict(self.request_context)
                approval_metadata.update({"tool_name": tool_name, "params": params})
                req = self.hitl.request_tool_approval(
                    tool_name,
                    params,
                    actor=actor,
                    risk="high",
                    metadata=approval_metadata,
                )
                if self.event_bus:
                    self.event_bus.emit(
                        "approval.requested",
                        actor,
                        {"approval": req.to_dict(), "tool": tool_name, "params": params},
                        request_id=self._request_id(),
                    )
                if req.status == "approved":
                    approved = True
                    if self.event_bus:
                        self.event_bus.emit(
                            "approval.auto_approved",
                            actor,
                            {"approval": req.to_dict(), "tool": tool_name},
                            request_id=self._request_id(),
                        )
                else:
                    if self.event_bus:
                        self.event_bus.emit(
                            "approval.pending" if req.status == "pending" else "approval.auto_denied",
                            actor,
                            {"approval": req.to_dict(), "tool": tool_name},
                            request_id=self._request_id(),
                        )
                    result = ToolResult(
                        False,
                        result={"status": req.status, "approval_id": req.approval_id, "preview": req.preview},
                        error=f"APPROVAL_PENDING: {req.approval_id}" if req.status == "pending" else f"Approval denied: {req.approval_id}",
                        tool_name=tool_name,
                    )
                    self._record(tool_name, params, result)
                    return result

        # Optional Notebook-24 policy layer.  Only enforce if policies were registered.
        if self.safety is not None and getattr(self.safety.tool_validator, "policies", None):
            allowed, validation = self.safety.check_tool_call(
                tool_name, params, actor=actor, approved=approved
            )
            if not allowed:
                result = ToolResult(False, error="; ".join(validation.get("issues", [])), tool_name=tool_name)
                self._record(tool_name, params, result)
                return result

        result = self.registry.call(tool_name, **params)
        self._record(tool_name, params, result)
        return result

    def _record(self, tool_name: str, params: dict, result: ToolResult) -> None:
        data = result.to_dict()
        record = ToolCallRecord(
            tool=tool_name,
            params=params,
            success=result.success,
            result=data.get("result"),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )
        self.call_records.append(record)
        if self.event_bus:
            self.event_bus.emit(
                "tool.call.completed",
                "tool_runtime",
                {"record": record.to_dict()},
                request_id=self._request_id(),
            )

    def stats(self) -> dict:
        return {
            "registry": self.registry.get_stats(),
            "records": len(self.call_records),
            "per_tool": self.registry.per_tool_stats(),
        }

    def recent_calls(self, limit: int = 20) -> list[dict]:
        return [r.to_dict() for r in self.call_records[-limit:]]

    def pending_approvals(self) -> list[dict]:
        if self.approval_store is None:
            return []
        return self.approval_store.pending_items()
