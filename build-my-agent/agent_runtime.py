"""Integrated Castalia AgentRuntime.

This is the trunk for the remaining notebooks: human-in-the-loop, evaluation,
optimization, resilience, MCP/A2A, lifecycle, and capstone orchestration should
extend this runtime rather than creating more isolated demo loops.
"""

from __future__ import annotations

from typing import Optional

from agent_adapters import build_default_adapters
from agent_safety import GuardrailsLayer, ToolPolicy
from approval_runtime import ApprovalStore, HumanInTheLoopController
from event_bus import EventBus
from memory_hub import MemoryHub
from optimization_runtime import RuntimeOptimizer
from registry_bootstrap import build_full_tool_registry
from runtime_contracts import AgentContext, AgentRequest, AgentResult
from strategy_router import StrategyRouter
from tool_runtime import ToolRuntime
from lifecycle_runtime import AgentLifecycleManager, AgentLogger, AgentRegistry


class AgentRuntime:
    """One public lifecycle for all agent strategies."""

    def __init__(
        self,
        tool_runtime: Optional[ToolRuntime] = None,
        memory_hub: Optional[MemoryHub] = None,
        safety: Optional[GuardrailsLayer] = None,
        event_bus: Optional[EventBus] = None,
        strategy_router: Optional[StrategyRouter] = None,
        adapters: Optional[dict] = None,
        runtime_config: Optional[dict] = None,
        approval_store: Optional[ApprovalStore] = None,
        optimizer: Optional[RuntimeOptimizer] = None,
    ):
        self.event_bus = event_bus or EventBus()
        self.tool_registry = build_full_tool_registry() if tool_runtime is None else tool_runtime.registry
        self.registry = self.tool_registry
        self.agent_logger = AgentLogger()
        self.agent_registry = AgentRegistry()
        self.lifecycle = AgentLifecycleManager(self.agent_registry, event_bus=self.event_bus, logger=self.agent_logger)
        self.safety = safety or GuardrailsLayer(agent_name="AgentRuntime", registry=self.tool_registry)
        self.approval_store = approval_store or ApprovalStore(mode="auto_deny")
        self.hitl = HumanInTheLoopController(self.approval_store)
        self._register_default_tool_policies()
        self.tool_runtime = tool_runtime or ToolRuntime(
            self.tool_registry,
            safety=self.safety,
            event_bus=self.event_bus,
            approval_store=self.approval_store,
        )
        # If a prebuilt ToolRuntime was injected, attach the runtime safety/event
        # layers unless the caller explicitly configured them.
        if getattr(self.tool_runtime, "safety", None) is None:
            self.tool_runtime.safety = self.safety
        if getattr(self.tool_runtime, "event_bus", None) is None:
            self.tool_runtime.event_bus = self.event_bus
        self.tool_runtime.approval_store = self.approval_store
        self.tool_runtime.hitl = self.hitl
        self.memory_hub = memory_hub or MemoryHub(short_strategy="sliding", window_size=12, enable_long_term=False)
        self.strategy_router = strategy_router or StrategyRouter()
        self.adapters = adapters or build_default_adapters()
        self.runtime_config = runtime_config or {"max_steps": 8, "verbose": True}
        self.optimizer = optimizer or RuntimeOptimizer(default_model=self.runtime_config.get("model"))

    def _register_default_tool_policies(self) -> None:
        """Whitelist all registered tools; mark confirmation as approval-gated."""
        for name in self.tool_registry.tools:
            requires_approval = name in {"confirm_action"}
            self.safety.register_tool_policy(ToolPolicy(
                name=name,
                enabled=True,
                requires_approval=requires_approval,
                max_calls_per_window=60,
                window_seconds=60,
            ))

    def register_agent(self, name: str, strategy: Optional[str] = None, description: str = "", capabilities: Optional[list] = None, config: Optional[dict] = None, auto_start: bool = True):
        strategy_name = strategy or name
        if strategy_name not in self.adapters:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        registration = self.agent_registry.register(
            name=name,
            strategy=strategy_name,
            description=description,
            capabilities=capabilities or [],
            config=config or {},
            handler=self.adapters[strategy_name],
        )
        self.lifecycle.register(name)
        if auto_start:
            self.lifecycle.start(name)
        return registration

    def start_agent(self, name: str) -> dict:
        return self.lifecycle.start(name)

    def stop_agent(self, name: str) -> dict:
        return self.lifecycle.stop(name)

    def restart_agent(self, name: str) -> dict:
        return self.lifecycle.restart(name)

    def health_check(self, name: str) -> dict:
        return self.lifecycle.health(name)

    def submit_task(self, agent_name: str, task: str, **kwargs) -> AgentResult:
        registration = self.agent_registry.get(agent_name)
        health = self.lifecycle.health(agent_name)
        if health["status"] != "running":
            return AgentResult(
                answer=f"Agent '{agent_name}' is not running (status: {health['status']}).",
                success=False,
                strategy_used=registration.strategy,
                errors=[f"Agent '{agent_name}' is not running"],
            ).finish()
        config = registration.config or {}
        merged_metadata = {**config.get("metadata", {}), **kwargs.pop("metadata", {}), "runtime_agent_name": agent_name}
        return self.run(
            AgentRequest(
                task=task,
                strategy=registration.strategy,
                max_steps=kwargs.pop("max_steps", config.get("max_steps")),
                user_id=kwargs.pop("user_id", "default"),
                session_id=kwargs.pop("session_id", "default"),
                metadata=merged_metadata,
            )
        )

    def get_status(self) -> dict:
        lifecycle = self.lifecycle.list_agents()
        return {
            "registered_agents": len(self.agent_registry),
            "running_agents": sum(1 for record in lifecycle if record["status"] == "running"),
            "agents": lifecycle,
            "event_stats": self.event_bus.stats(),
            "log_stats": self.agent_logger.stats(),
        }

    def run(self, request_or_task, **kwargs) -> AgentResult:
        """Run a task through the full lifecycle.

        Accepts either an AgentRequest or a raw task string.
        """
        request = request_or_task if isinstance(request_or_task, AgentRequest) else AgentRequest(task=str(request_or_task), **kwargs)
        self.event_bus.emit("request.started", "runtime", {"request": request.to_dict()}, request_id=request.request_id)

        # Input guardrails.
        safe, input_result = self.safety.check_input(request.task, actor=request.user_id)
        if not safe:
            result = AgentResult(
                answer="Request blocked: potentially unsafe input detected.",
                success=False,
                strategy_used="blocked",
                errors=[str(input_result.get("issues", []))],
                request_id=request.request_id,
            ).finish()
            self.event_bus.emit("request.blocked", "runtime", result.to_dict(), request_id=request.request_id)
            return result

        # Request rate limit.
        allowed, reason = self.safety.request_limiter.is_allowed(request.user_id)
        if not allowed:
            result = AgentResult(
                answer="Request blocked: rate limit exceeded.",
                success=False,
                strategy_used="rate_limited",
                errors=[reason],
                request_id=request.request_id,
            ).finish()
            self.event_bus.emit("request.blocked", "runtime", result.to_dict(), request_id=request.request_id)
            return result

        # Memory recall; inject compact context into task for adapters that do not yet
        # accept a separate memory prompt.
        memory_context = self.memory_hub.recall(request.task, k=5)
        effective_request = request
        if memory_context:
            memory_text = "\n\n".join(m["content"] for m in memory_context)
            effective_request = AgentRequest(
                task=f"Relevant prior context:\n{memory_text}\n\nCurrent task:\n{request.task}",
                user_id=request.user_id,
                session_id=request.session_id,
                strategy=request.strategy,
                max_steps=request.max_steps,
                metadata={**request.metadata, "original_task": request.task, "memory_injected": True},
                request_id=request.request_id,
            )

        # Select strategy.
        strategy = self.strategy_router.select(request.task, request.strategy)
        adapter = self.adapters.get(strategy)
        if adapter is None:
            adapter = self.adapters.get("react")
            strategy = "react"

        self.event_bus.emit("strategy.selected", "runtime", {"strategy": strategy}, request_id=request.request_id)
        runtime_agent_name = request.metadata.get("runtime_agent_name", strategy)
        self.agent_logger.log(runtime_agent_name, "INFO", "Strategy selected", {"request_id": request.request_id, "strategy": strategy})

        context = AgentContext(
            request=effective_request,
            memory_context=memory_context,
            tool_runtime=self.tool_runtime,
            memory_hub=self.memory_hub,
            event_bus=self.event_bus,
            optimizer=self.optimizer,
            runtime_config=self.runtime_config,
        )

        self.tool_runtime.bind_request_context(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            strategy=strategy,
            task=request.task,
            metadata=request.metadata,
        )
        try:
            result = adapter.run(effective_request, context)
            result.request_id = request.request_id
            self.agent_logger.log(runtime_agent_name, "INFO", "Request completed", {"request_id": request.request_id, "success": result.success, "strategy": strategy})
        except Exception as e:
            self.agent_logger.log(runtime_agent_name, "ERROR", "Request failed", {"request_id": request.request_id, "error": f"{type(e).__name__}: {e}", "strategy": strategy})
            if runtime_agent_name in self.agent_registry:
                self.lifecycle.mark_error(runtime_agent_name, f"{type(e).__name__}: {e}")
            result = AgentResult(
                answer=f"Agent failed: {type(e).__name__}: {e}",
                success=False,
                strategy_used=strategy,
                errors=[f"{type(e).__name__}: {e}"],
                request_id=request.request_id,
            ).finish()
        finally:
            self.tool_runtime.clear_request_context()

        escalation = self.hitl.maybe_escalate(request.task, result=result)
        if escalation is not None:
            result.metadata["escalation"] = escalation.to_dict()
            self.event_bus.emit("escalation.created", "runtime", escalation.to_dict(), request_id=request.request_id)

        # Output guardrails.
        filtered, scan = self.safety.filter_output(result.answer, actor="AgentRuntime")
        if filtered != result.answer:
            result.artifacts["unfiltered_answer"] = result.answer
            result.answer = filtered
            result.metadata["output_redacted"] = True
            result.metadata["output_filter_findings"] = scan.get("findings", [])

        # Memory writeback.
        self.memory_hub.store_result(request, result)
        if runtime_agent_name in self.agent_registry:
            if result.success:
                self.lifecycle.mark_task_completed(runtime_agent_name)
            else:
                error_text = "; ".join(result.errors) if result.errors else "request unsuccessful"
                self.lifecycle.mark_error(runtime_agent_name, error_text)
        self.event_bus.emit("request.completed", "runtime", result.to_dict(), request_id=request.request_id)
        return result.finish()

    def stats(self) -> dict:
        return {
            "tools": self.tool_runtime.stats(),
            "memory": self.memory_hub.stats(),
            "events": self.event_bus.stats(),
            "safety": self.safety.audit.summary(),
            "adapters": sorted(self.adapters.keys()),
            "hitl": self.hitl.stats(),
            "optimization": self.optimizer.stats(),
            "agent_runtime": self.get_status(),
        }

    def pending_approvals(self) -> list[dict]:
        return self.approval_store.pending_items()

    def decide_approval(self, approval_id: str, approved: bool, reason: str = "") -> dict:
        decision = self.approval_store.decide(approval_id, approved=approved, reason=reason)
        event_type = "approval.approved" if approved else "approval.denied"
        self.event_bus.emit("approval.decided", "runtime", decision)
        self.event_bus.emit(event_type, "runtime", decision)
        return decision

    def resume_approval(self, approval_id: str, actor: str = "human") -> AgentResult:
        decision = self.approval_store.decisions.get(approval_id)
        if not decision:
            return AgentResult(
                answer=f"Unknown approval id: {approval_id}",
                success=False,
                strategy_used="approval_resume",
                errors=[f"Unknown approval id: {approval_id}"],
            ).finish()
        if not decision.get("approved"):
            return AgentResult(
                answer=f"Approval {approval_id} was denied. Pending tool action will not run.",
                success=False,
                strategy_used="approval_resume",
                metadata={"approval_id": approval_id, "decision": decision},
            ).finish()

        approval_request = decision["request"]
        before = len(self.tool_runtime.call_records)
        self.tool_runtime.bind_request_context(**(approval_request.get("metadata") or {}))
        try:
            tool_result = self.tool_runtime.call(
                approval_request["tool_name"],
                approval_request["params"],
                actor=actor,
                approved=True,
            )
        finally:
            self.tool_runtime.clear_request_context()
        tool_calls = [r.to_dict() for r in self.tool_runtime.call_records[before:]]
        answer = self.hitl.summarize_tool_result(approval_request["tool_name"], tool_result)
        result = AgentResult(
            answer=answer,
            success=tool_result.success,
            strategy_used="approval_resume",
            tool_calls=tool_calls,
            artifacts={"tool_result": tool_result.to_dict()},
            metadata={"approval_id": approval_id, "decision": decision, "approval_request": approval_request},
        ).finish()
        filtered, scan = self.safety.filter_output(result.answer, actor="AgentRuntime")
        if filtered != result.answer:
            result.artifacts["unfiltered_answer"] = result.answer
            result.answer = filtered
            result.metadata["output_redacted"] = True
            result.metadata["output_filter_findings"] = scan.get("findings", [])
        self.event_bus.emit(
            "approval.resumed",
            actor,
            {"approval_id": approval_id, "tool_result": tool_result.to_dict()},
            request_id=(approval_request.get("metadata") or {}).get("request_id"),
        )
        self.memory_hub.store_result(
            AgentRequest(task=approval_request.get("action", "resume approval"), user_id=actor, strategy="approval_resume"),
            result,
        )
        return result

    def revise_with_feedback(self, task: str, draft: str, feedback: str, actor: str = "human") -> AgentResult:
        revised = self.hitl.revise_with_feedback(task, draft, feedback)
        feedback_record = revised["feedback"]
        self.event_bus.emit("feedback.recorded", actor, feedback_record.to_dict())
        result = AgentResult(
            answer=revised["revised_draft"],
            success=bool(revised["revised_draft"].strip()),
            strategy_used="feedback_loop",
            artifacts={"original_draft": draft, "feedback_record": feedback_record.to_dict()},
            metadata={"feedback_id": feedback_record.feedback_id, "task": task},
        ).finish()
        filtered, scan = self.safety.filter_output(result.answer, actor="AgentRuntime")
        if filtered != result.answer:
            result.artifacts["unfiltered_answer"] = result.answer
            result.answer = filtered
            result.metadata["output_redacted"] = True
            result.metadata["output_filter_findings"] = scan.get("findings", [])
        self.event_bus.emit("feedback.revision.completed", actor, result.to_dict())
        self.memory_hub.store_result(
            AgentRequest(task=task, user_id=actor, strategy="feedback_loop"),
            result,
        )
        return result

    def interactive_trace(self, request_id: str, limit: int = 200) -> list[dict]:
        return self.event_bus.recent(limit=limit, request_id=request_id)

    def evaluate_dataset(self, dataset=None, judge_fn=None, use_judge: bool = True, name: Optional[str] = None):
        from evaluation_runtime import AgentEvaluator, build_default_golden_dataset

        evaluator = AgentEvaluator(self, name=name or "AgentRuntime", judge_fn=judge_fn)
        tasks = dataset or build_default_golden_dataset()
        evaluator.evaluate_dataset(tasks, use_judge=use_judge)
        return evaluator

    def evaluate_default_dataset(self, judge_fn=None, use_judge: bool = True) -> dict:
        evaluator = self.evaluate_dataset(judge_fn=judge_fn, use_judge=use_judge)
        return evaluator.aggregate_report()


def build_default_runtime(**config) -> AgentRuntime:
    runtime_config = {"max_steps": 8, "verbose": True}
    runtime_config.update(config)
    return AgentRuntime(runtime_config=runtime_config)


if __name__ == "__main__":
    runtime = build_default_runtime(max_steps=3)
    result = runtime.run("Use the calculator tool to compute 2 + 3 * 4, then answer with the result.")
    print(result.to_dict())
