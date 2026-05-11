"""Adapters that normalize existing notebook agents into AgentResult."""

from __future__ import annotations

import time
from typing import Dict, Optional

from runtime_contracts import AgentContext, AgentRequest, AgentResult


class AgentAdapter:
    name = "base"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        raise NotImplementedError


class RegistryReactAdapter(AgentAdapter):
    """Default runtime agent: ReAct-style loop over the full ToolRegistry."""

    name = "react"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        from agent_tool_integration import AdvancedToolAgent

        max_steps = request.max_steps or context.runtime_config.get("max_steps", 8)
        agent = AdvancedToolAgent(
            context.tool_runtime,
            max_steps=max_steps,
            event_bus=context.event_bus,
            request_id=request.request_id,
            actor=request.user_id,
            optimizer=context.optimizer,
        )
        raw = agent.run(request.task, verbose=context.runtime_config.get("verbose", True))
        tool_calls = context.tool_runtime.recent_calls(raw.get("tool_calls", 20))
        pending_approval = any(str(tc.get("error", "")).startswith("APPROVAL_PENDING:") for tc in tool_calls)
        denied_approval = any("Approval denied" in str(tc.get("error", "")) for tc in tool_calls)
        completed = not str(raw.get("answer", "")).startswith("Max steps") and not pending_approval and not denied_approval
        return AgentResult(
            answer=raw.get("answer", ""),
            success=completed,
            strategy_used=self.name,
            steps=raw.get("steps", []),
            tool_calls=tool_calls,
            artifacts={"raw": raw},
            metadata={
                "total_time": raw.get("total_time"),
                "tool_calls": raw.get("tool_calls"),
                "input_tokens": raw.get("input_tokens", 0),
                "output_tokens": raw.get("output_tokens", 0),
                "total_tokens": raw.get("total_tokens", 0),
                "requires_human_approval": pending_approval,
                "approval_denied": denied_approval,
                "optimization": raw.get("optimization", {}),
            },
            request_id=request.request_id,
            started_at=time.time() - raw.get("total_time", 0),
        ).finish()


class AgentLoopV2Adapter(AgentAdapter):
    name = "loop_v2"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        from agent_loop_v2 import AgentLoopV2

        agent = AgentLoopV2(
            task=request.task,
            tool_runtime=context.tool_runtime,
            max_steps=request.max_steps or context.runtime_config.get("max_steps", 8),
            window_size=context.runtime_config.get("memory_window", 12),
        )
        result = agent.run()
        result.request_id = request.request_id
        return result


class LegacyLoopAdapter(AgentAdapter):
    name = "legacy_loop"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        from agent_loop import AgentLoop

        agent = AgentLoop(goal=request.task, max_steps=request.max_steps or 10)
        state = agent.run()
        return AgentResult(
            answer=state.final_answer,
            success=state.is_complete,
            strategy_used=self.name,
            steps=state.steps,
            tool_calls=[],
            artifacts={"state": state.summary()},
            metadata=state.metadata,
            request_id=request.request_id,
        ).finish()


class ReActAdapter(AgentAdapter):
    """Adapter around the original educational ReActAgent."""

    name = "legacy_react"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        from react_agent import ReActAgent

        agent = ReActAgent(max_steps=request.max_steps or context.runtime_config.get("max_steps", 10))
        raw = agent.run(request.task)
        return AgentResult(
            answer=raw.get("answer") or "",
            success=raw.get("is_complete", False),
            strategy_used=self.name,
            steps=raw.get("cycles", []),
            artifacts={"raw": raw},
            metadata={"time": raw.get("time"), "steps": raw.get("steps")},
            request_id=request.request_id,
        ).finish()


class PlanExecuteAdapter(AgentAdapter):
    name = "plan_execute"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        from plan_agent import PlanAndExecuteAgent

        agent = PlanAndExecuteAgent(max_steps_per_subtask=context.runtime_config.get("max_steps_per_subtask", 8))
        raw = agent.run(request.task)
        completed = len(raw.get("completed_steps", []))
        total = raw.get("total_steps", completed)
        return AgentResult(
            answer=raw.get("final_answer", ""),
            success=completed == total,
            strategy_used=self.name,
            steps=list(raw.get("results", {}).values()),
            artifacts={"plan": raw.get("plan"), "raw": raw},
            metadata={"completed_steps": completed, "total_steps": total},
            request_id=request.request_id,
        ).finish()


class ReflectionAdapter(AgentAdapter):
    name = "reflection"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        from plan_agent import PlanAndExecuteAgent
        from reflection_agent import ReflectionAgent

        inner = PlanAndExecuteAgent(max_steps_per_subtask=context.runtime_config.get("max_steps_per_subtask", 8))
        agent = ReflectionAgent(
            inner_agent=inner,
            threshold=context.runtime_config.get("reflection_threshold", 7.0),
            max_iterations=context.runtime_config.get("reflection_iterations", 3),
        )
        raw = agent.run(request.task)
        return AgentResult(
            answer=raw.get("final_answer", ""),
            success=True,
            strategy_used=self.name,
            steps=raw.get("reflection_history", []),
            artifacts={"raw": raw, "inner": raw.get("inner_agent_result")},
            metadata={"iterations": raw.get("iterations"), "threshold": raw.get("threshold")},
            request_id=request.request_id,
        ).finish()


class CodeAgentAdapter(AgentAdapter):
    name = "code"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        from code_agent import CodeAgent
        from code_executor import CodeExecutor

        executor = CodeExecutor(timeout=context.runtime_config.get("code_timeout", 30))
        agent = CodeAgent(executor=executor, max_cycles=context.runtime_config.get("code_max_cycles", 5))
        state = agent.run(request.task, user_id=request.user_id)
        return AgentResult(
            answer=state.final_answer,
            success=state.is_complete and getattr(state, "success_cycles", 0) > 0,
            strategy_used=self.name,
            steps=state.cycles,
            artifacts={"state_summary": state.summary() if hasattr(state, "summary") else {}},
            metadata={"cycles": state.current_cycle, "executor": executor.get_stats()},
            request_id=request.request_id,
        ).finish()


class ResearchAdapter(AgentAdapter):
    name = "research"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        from tavily_research_agent import TavilyResearchAgent

        agent = TavilyResearchAgent()
        report = agent.research(request.task)
        answer = getattr(report, "answer", None) or str(report)
        return AgentResult(
            answer=answer,
            success=True,
            strategy_used=self.name,
            artifacts={"report": report},
            request_id=request.request_id,
        ).finish()


class DataAgentAdapter(AgentAdapter):
    """For now, route data tasks through the registry ReAct agent with data tools."""

    name = "data"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        return RegistryReactAdapter().run(request, context)


class MCPReactAdapter(AgentAdapter):
    """ReAct over tools discovered dynamically from configured MCP servers."""

    name = "mcp"

    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        from agent_tool_integration import AdvancedToolAgent
        from protocol_runtime import build_registry_from_mcp_servers
        from tool_runtime import ToolRuntime

        servers = (
            request.metadata.get("mcp_servers")
            or context.runtime_config.get("mcp_servers")
            or []
        )
        if not servers:
            return AgentResult(
                answer="No MCP servers configured. Provide runtime_config['mcp_servers'] or request.metadata['mcp_servers'].",
                success=False,
                strategy_used=self.name,
                errors=["No MCP servers configured"],
                request_id=request.request_id,
            ).finish()

        cache_key = tuple(getattr(server, "name", repr(server)) for server in servers)
        cache = context.runtime_config.setdefault("_mcp_registry_cache", {})
        bundle = cache.get(cache_key)
        if bundle is None:
            bundle = build_registry_from_mcp_servers(list(servers))
            cache[cache_key] = bundle
            if context.event_bus:
                context.event_bus.emit(
                    "mcp.discovery.completed",
                    request.user_id,
                    {"servers": sorted(bundle.clients.keys()), "tools": sorted(bundle.tool_index.keys())},
                    request_id=request.request_id,
                )

        tool_runtime = ToolRuntime(
            bundle.registry,
            safety=context.tool_runtime.safety,
            event_bus=context.event_bus,
            approval_store=context.tool_runtime.approval_store,
        )
        safety = context.tool_runtime.safety
        if safety is not None and getattr(safety.tool_validator, "policies", None) is not None:
            from agent_safety import ToolPolicy

            for tool_name in bundle.registry.tools:
                if tool_name not in safety.tool_validator.policies:
                    safety.register_tool_policy(ToolPolicy(
                        name=tool_name,
                        enabled=True,
                        requires_approval=False,
                        max_calls_per_window=60,
                        window_seconds=60,
                    ))
        tool_runtime.bind_request_context(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            strategy=self.name,
            task=request.task,
            metadata={**request.metadata, "mcp_servers": [getattr(s, 'name', str(s)) for s in servers]},
        )
        try:
            max_steps = request.max_steps or context.runtime_config.get("max_steps", 8)
            agent = AdvancedToolAgent(
                tool_runtime,
                max_steps=max_steps,
                event_bus=context.event_bus,
                request_id=request.request_id,
                actor=request.user_id,
                optimizer=context.optimizer,
            )
            raw = agent.run(request.task, verbose=context.runtime_config.get("verbose", True))
        finally:
            tool_runtime.clear_request_context()

        tool_calls = tool_runtime.recent_calls(raw.get("tool_calls", 20))
        pending_approval = any(str(tc.get("error", "")).startswith("APPROVAL_PENDING:") for tc in tool_calls)
        denied_approval = any("Approval denied" in str(tc.get("error", "")) for tc in tool_calls)
        completed = not str(raw.get("answer", "")).startswith("Max steps") and not pending_approval and not denied_approval
        return AgentResult(
            answer=raw.get("answer", ""),
            success=completed,
            strategy_used=self.name,
            steps=raw.get("steps", []),
            tool_calls=tool_calls,
            artifacts={
                "raw": raw,
                "mcp_servers": sorted(bundle.clients.keys()),
                "mcp_tools": sorted(bundle.tool_index.keys()),
            },
            metadata={
                "total_time": raw.get("total_time"),
                "tool_calls": raw.get("tool_calls"),
                "input_tokens": raw.get("input_tokens", 0),
                "output_tokens": raw.get("output_tokens", 0),
                "total_tokens": raw.get("total_tokens", 0),
                "requires_human_approval": pending_approval,
                "approval_denied": denied_approval,
                "optimization": raw.get("optimization", {}),
                "mcp_servers": sorted(bundle.clients.keys()),
                "mcp_tools": sorted(bundle.tool_index.keys()),
            },
            request_id=request.request_id,
            started_at=time.time() - raw.get("total_time", 0),
        ).finish()


def build_default_adapters() -> Dict[str, AgentAdapter]:
    adapters = [
        RegistryReactAdapter(),
        AgentLoopV2Adapter(),
        LegacyLoopAdapter(),
        ReActAdapter(),
        PlanExecuteAdapter(),
        ReflectionAdapter(),
        CodeAgentAdapter(),
        ResearchAdapter(),
        DataAgentAdapter(),
        MCPReactAdapter(),
    ]
    # Aliases preserve old names while steering default users to registry-backed ReAct.
    mapping = {a.name: a for a in adapters}
    mapping["auto"] = mapping["react"]
    mapping["loop"] = mapping["loop_v2"]
    return mapping
