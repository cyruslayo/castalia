from approval_runtime import ApprovalStore, FeedbackRecord, HumanInTheLoopController
from agent_runtime import AgentRuntime
from agent_safety import GuardrailsLayer, ToolPolicy
from event_bus import EventBus
from runtime_contracts import AgentContext, AgentRequest, AgentResult
from tool_registry import build_production_registry
from tool_runtime import ToolRuntime


def test_safe_printable_handles_unicode_for_windows_console():
    from agent_tool_integration import _safe_printable

    text = _safe_printable("approved ✅")
    assert isinstance(text, str)
    assert "approved" in text


def test_approval_store_manual_lifecycle():
    store = ApprovalStore(mode="manual")
    req = store.request(
        action="Send email",
        tool_name="send_email",
        params={"to": "user@example.com"},
        preview="send_email({'to': 'user@example.com'})",
        risk="high",
    )

    assert req.status == "pending"
    assert len(store.pending_items()) == 1

    decision = store.decide(req.approval_id, approved=True, reason="looks safe")
    assert decision["success"] is True
    assert decision["approved"] is True
    assert store.is_approved(req.approval_id) is True
    assert store.stats()["pending"] == 0


def test_tool_runtime_creates_pending_approval_for_confirm_action():
    registry = build_production_registry()
    safety = GuardrailsLayer(agent_name="TestRuntime", registry=registry)
    store = ApprovalStore(mode="manual")
    runtime = ToolRuntime(registry, safety=safety, approval_store=store)

    safety.register_tool_policy(ToolPolicy("request_delete", requires_approval=False))
    safety.register_tool_policy(ToolPolicy("confirm_action", requires_approval=True))

    request_delete = runtime.call("request_delete", {"resource_type": "file", "resource_id": "draft.txt"})
    token = request_delete.result["token"]

    pending = runtime.call("confirm_action", {"token": token})
    assert pending.success is False
    assert pending.error.startswith("APPROVAL_PENDING:")
    assert pending.to_dict()["result"]["status"] == "pending"
    assert len(runtime.pending_approvals()) == 1


def test_hitl_controller_feedback_revision_uses_stub_llm():
    hitl = HumanInTheLoopController()
    revised = hitl.revise_with_feedback(
        task="Write a note",
        draft="This is a long note.",
        feedback="Make it shorter.",
        llm_fn=lambda messages: "Short note.",
    )

    assert revised["revised_draft"] == "Short note."
    assert revised["feedback"].feedback == "Make it shorter."
    assert len(hitl.feedback_log) == 1


def test_hitl_controller_skips_escalation_for_pending_approval():
    hitl = HumanInTheLoopController()
    result = AgentResult(
        answer="Waiting for approval",
        success=False,
        strategy_used="react",
        metadata={"requires_human_approval": True},
    )
    assert hitl.maybe_escalate("delete a record", result=result) is None


def test_registry_react_adapter_marks_pending_approval_unsuccessful(monkeypatch):
    import agent_tool_integration
    from agent_adapters import RegistryReactAdapter

    class FakeAgent:
        def __init__(self, tool_backend, max_steps=8, **kwargs):
            self.tool_backend = tool_backend
            self.max_steps = max_steps

        def run(self, query: str, verbose: bool = True):
            return {"answer": "Approval is pending.", "steps": [], "tool_calls": 1, "total_time": 0.0}

    monkeypatch.setattr(agent_tool_integration, "AdvancedToolAgent", FakeAgent)

    registry = build_production_registry()
    tool_runtime = ToolRuntime(registry)
    tool_runtime.call_records.append(type("R", (), {"to_dict": lambda self: {
        "tool": "confirm_action",
        "params": {"token": "confirm_1"},
        "success": False,
        "result": {"status": "pending"},
        "error": "APPROVAL_PENDING: approval-123",
        "execution_time_ms": 0.0,
        "call_id": "tool-1",
    }})())
    context = AgentContext(request=AgentRequest(task="delete"), tool_runtime=tool_runtime, runtime_config={})

    result = RegistryReactAdapter().run(AgentRequest(task="delete"), context)
    assert result.success is False
    assert result.metadata["requires_human_approval"] is True


def test_agent_runtime_resume_approval_executes_tool():
    registry = build_production_registry()
    safety = GuardrailsLayer(agent_name="AgentRuntime", registry=registry)
    tool_runtime = ToolRuntime(registry, safety=safety, approval_store=ApprovalStore(mode="manual"))
    runtime = AgentRuntime(tool_runtime=tool_runtime, safety=safety, approval_store=tool_runtime.approval_store)

    request_delete = runtime.tool_runtime.call("request_delete", {"resource_type": "record", "resource_id": "123"})
    token = request_delete.result["token"]
    blocked = runtime.tool_runtime.call("confirm_action", {"token": token})
    assert blocked.success is False

    approval_id = runtime.pending_approvals()[0]["approval_id"]
    runtime.decide_approval(approval_id, approved=True, reason="approved in test")
    resumed = runtime.resume_approval(approval_id)

    assert resumed.success is True
    assert resumed.metadata["approval_id"] == approval_id
    assert resumed.tool_calls[0]["tool"] == "confirm_action"
    assert resumed.tool_calls[0]["result"]["resource_id"] == "123"


def test_agent_runtime_revise_with_feedback_emits_events(monkeypatch):
    runtime = AgentRuntime(runtime_config={"max_steps": 1, "verbose": False}, approval_store=ApprovalStore(mode="manual"))

    def fake_revise(task, draft, feedback):
        return {
            "feedback": FeedbackRecord(task=task, draft=draft, feedback=feedback),
            "revised_draft": "Revised draft.",
        }

    monkeypatch.setattr(runtime.hitl, "revise_with_feedback", fake_revise)
    result = runtime.revise_with_feedback("Write a memo", "Draft", "Be shorter")

    assert result.success is True
    assert result.answer == "Revised draft."
    types = [e["event_type"] for e in runtime.event_bus.recent(limit=10)]
    assert "feedback.recorded" in types
    assert "feedback.revision.completed" in types


def test_interactive_trace_filters_by_request_id():
    bus = EventBus()
    bus.emit("agent.step.parsed", "agent", {"step": 1}, request_id="req-1")
    bus.emit("agent.step.parsed", "agent", {"step": 2}, request_id="req-2")
    bus.emit("tool.call.completed", "tool_runtime", {"step": 3}, request_id="req-1")
    runtime = AgentRuntime(event_bus=bus, approval_store=ApprovalStore(mode="manual"))

    trace = runtime.interactive_trace("req-1")
    assert len(trace) == 2
    assert all(item["request_id"] == "req-1" for item in trace)


def test_agent_runtime_manual_approval_round_trip():
    registry = build_production_registry()
    safety = GuardrailsLayer(agent_name="AgentRuntime", registry=registry)
    tool_runtime = ToolRuntime(registry, safety=safety, approval_store=ApprovalStore(mode="manual"))
    runtime = AgentRuntime(tool_runtime=tool_runtime, safety=safety, approval_store=tool_runtime.approval_store)

    request_delete = runtime.tool_runtime.call("request_delete", {"resource_type": "record", "resource_id": "123"})
    token = request_delete.result["token"]

    blocked = runtime.tool_runtime.call("confirm_action", {"token": token})
    assert blocked.success is False
    approval_id = runtime.pending_approvals()[0]["approval_id"]

    decision = runtime.decide_approval(approval_id, approved=True, reason="approved in test")
    assert decision["success"] is True
    assert decision["approved"] is True

    confirmed = runtime.tool_runtime.call("confirm_action", {"token": token}, approved=True)
    assert confirmed.success is True
    assert confirmed.result["resource_id"] == "123"
