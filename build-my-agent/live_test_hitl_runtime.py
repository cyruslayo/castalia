"""Live Notebook 25 validation against the configured OpenAI-compatible LLM.

Covers:
  1. Manual approval gate with real tool-using agent
  2. Approval resume path after human approval
  3. Auto-allow approval mode
  4. Feedback-driven revision with a real LLM call
  5. Interactive trace / event visibility

Usage:
    cd build-my-agent && python live_test_hitl_runtime.py
"""

from __future__ import annotations

from openai import OpenAI

import config
from agent_tool_integration import _safe_printable
from approval_runtime import ApprovalStore
from agent_runtime import AgentRuntime


DELETE_TASK = "Delete the record with id customer-123 using the available tools. If approval is required, explain exactly what is pending."


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_long_timeout_client() -> None:
    config.client = OpenAI(
        base_url=config.LLM_CONFIG["base_url"],
        api_key=config.LLM_CONFIG["api_key"],
        timeout=600,
    )


def run_manual_approval_test() -> tuple[AgentRuntime, object, str, str]:
    runtime = AgentRuntime(
        runtime_config={"max_steps": 6, "verbose": True},
        approval_store=ApprovalStore(mode="manual"),
    )
    result = runtime.run(DELETE_TASK, strategy="react", user_id="live-hitl")

    print("\n[manual] success:", result.success)
    print("[manual] metadata:", _safe_printable(result.metadata))
    print("[manual] answer:", _safe_printable(result.answer))

    check(result.success is False, "manual approval run should pause, not succeed")
    check(result.metadata.get("requires_human_approval") is True, "manual run should require human approval")
    check("escalation" not in result.metadata, "manual approval pause should not be mislabeled as escalation")
    check(len(runtime.pending_approvals()) == 1, "manual run should create exactly one pending approval")

    approval = runtime.pending_approvals()[0]
    approval_id = approval["approval_id"]
    trace = runtime.interactive_trace(result.request_id)
    event_types = [evt["event_type"] for evt in trace]
    print("[manual] events:", _safe_printable(event_types))

    check("agent.run.started" in event_types, "interactive trace should include agent start")
    check("agent.step.parsed" in event_types, "interactive trace should include parsed agent steps")
    check("tool.call.requested" in event_types, "interactive trace should include tool requests")
    check("approval.requested" in event_types, "interactive trace should include approval requests")
    check("approval.pending" in event_types, "interactive trace should include approval pending")

    decision = runtime.decide_approval(approval_id, approved=True, reason="approved in live test")
    check(decision["success"] is True and decision["approved"] is True, "approval decision should succeed")

    resumed = runtime.resume_approval(approval_id)
    print("[manual->resume] success:", resumed.success)
    print("[manual->resume] answer:", _safe_printable(resumed.answer))
    print("[manual->resume] tool_calls:", _safe_printable(resumed.tool_calls))

    check(resumed.success is True, "resumed approval should execute pending tool successfully")
    check(resumed.tool_calls and resumed.tool_calls[0]["tool"] == "confirm_action", "resume should execute confirm_action")
    check(resumed.tool_calls[0]["result"]["resource_id"] == "customer-123", "resume should delete target record")

    return runtime, result, approval_id, result.request_id


def run_auto_allow_test() -> None:
    runtime = AgentRuntime(
        runtime_config={"max_steps": 6, "verbose": True},
        approval_store=ApprovalStore(mode="auto_allow"),
    )
    result = runtime.run(DELETE_TASK, strategy="react", user_id="live-hitl")

    print("\n[auto_allow] success:", result.success)
    print("[auto_allow] metadata:", _safe_printable(result.metadata))
    print("[auto_allow] answer:", _safe_printable(result.answer))

    check(result.success is True, "auto_allow run should succeed")
    check(not runtime.pending_approvals(), "auto_allow run should not leave pending approvals")
    trace = runtime.interactive_trace(result.request_id)
    event_types = [evt["event_type"] for evt in trace]
    check("approval.auto_approved" in event_types, "auto_allow run should emit auto approval event")


def run_feedback_test() -> None:
    runtime = AgentRuntime(runtime_config={"max_steps": 4, "verbose": False}, approval_store=ApprovalStore(mode="manual"))
    draft = (
        "Castalia Scholar helps analysts synthesize sources, draft reports, and maintain audit trails across tasks. "
        "It can support evidence-heavy workflows in enterprise teams."
    )
    feedback = "Revise this into a single concise sentence and explicitly mention approvals."
    result = runtime.revise_with_feedback("Write a concise product blurb", draft, feedback)

    print("\n[feedback] success:", result.success)
    print("[feedback] answer:", _safe_printable(result.answer))

    check(result.success is True, "feedback revision should succeed")
    check("approval" in result.answer.lower(), "revised draft should mention approvals")
    check(result.metadata.get("feedback_id"), "feedback revision should record feedback id")


def main() -> None:
    build_long_timeout_client()
    runtime, result, approval_id, request_id = run_manual_approval_test()
    run_auto_allow_test()
    run_feedback_test()
    print("\n[SUCCESS] Live Notebook 25 HITL runtime validation passed")
    print(f"  request_id={request_id}")
    print(f"  approval_id={approval_id}")


if __name__ == "__main__":
    main()
