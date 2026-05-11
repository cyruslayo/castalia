# Integrated Runtime Wiring

This document records the implementation trunk for wiring the notebook modules into one complete system.

## New public entry point

```python
from agent_runtime import build_default_runtime

runtime = build_default_runtime(max_steps=8, verbose=True)
result = runtime.run("Use tools to solve this task")
print(result.answer)
```

For explicit strategy selection:

```python
from runtime_contracts import AgentRequest

result = runtime.run(AgentRequest(
    task="Research current sources and cite them",
    strategy="research",
    user_id="alice",
))
```

## Core files

- `runtime_contracts.py` — `AgentRequest`, `AgentResult`, `AgentContext`, events and tool-call records.
- `agent_runtime.py` — lifecycle: input safety → rate limit → memory recall → strategy routing → adapter execution → output filtering → memory writeback. Also exposes `evaluate_dataset()` / `evaluate_default_dataset()` so Notebook 26 evaluation runs against the canonical runtime.
- `registry_bootstrap.py` — full registry builder: production tools + secure code executor + data tools + live/local/search tools.
- `tool_runtime.py` — canonical dispatch wrapper around `ToolRegistry` with safety and events.
- `agent_loop_v2.py` — registry-backed successor to the original `AgentLoop`.
- `agent_adapters.py` — normalizes existing agents into `AgentResult`.
- `strategy_router.py` — heuristic `auto` routing.
- `memory_hub.py` — short-term plus optional long-term/graph memory facade.
- `event_bus.py` — in-process structured events with pub/sub subscriptions, request-scoped history, and event stats.

## Production/notebook foundations

- `blackboard_runtime.py` — Notebook 22/capstone shared state facade.
- `approval_runtime.py` — Notebook 25 human-in-the-loop primitives: approval store, feedback revision, escalation records, tool-result summarisation, and runtime-facing HITL controller.
- `evaluation_runtime.py` — Notebook 26 evaluation harness: golden tasks, scoring helpers, `AgentEvaluator`, cost tracking, and regression baselines.
- `optimization_runtime.py` — Notebook 27 cache/token/model-routing primitives, plus `RuntimeOptimizer` now wired into the default registry-backed runtime path.
- `resilience.py` — Notebook 28 resilience stack: retry-with-feedback, fallback chain, circuit breaker, graceful degradation, cross-platform timeout helpers, and `SafeToolExecutor`.
- `protocol_runtime.py` — Notebook 29 implementation of local MCP (JSON-RPC messages, MCP server/client, tool schemas, runtime bridge, multi-server agent, MCP-to-ToolRegistry discovery bridge) plus Notebook 30 A2A runtime pieces (`AgentCard`, `AgentDirectory`, `TaskRequest`/`TaskResponse`, `A2AAgent`, `CoordinatorAgent`, capability discovery, local task dispatch).
- `lifecycle_runtime.py` — Notebook 31 runtime infrastructure: `AgentLogger`, logical `AgentRegistry`, and `AgentLifecycleManager` (register/start/stop/restart/health/trace).
- `capstone_runtime.py` — Notebook 32-37 Castalia Scholar orchestration scaffold.

## Canonical tool registry

Use `build_full_tool_registry()` for all new work. It includes:

- Base/prod/stateful tools from `tool_registry.py` and `production_tools.py`
- Secure code execution via `CodeExecutor` (`execute_python_secure`, and `python` override)
- Tavily `web_search` when configured
- local TF-IDF, semantic, and hybrid search wrappers
- schema-aware filesystem and unified data tools

## Strategy names

- `react` — registry-backed ReAct loop (`AdvancedToolAgent`) and default runtime strategy
- `loop_v2` / `loop` — registry-backed successor to original `AgentLoop`
- `legacy_loop` — original `AgentLoop`
- `legacy_react` — original educational `ReActAgent`
- `plan_execute` — `PlanAndExecuteAgent`
- `reflection` — `ReflectionAgent` over plan-execute
- `code` — `CodeAgent`
- `research` — `TavilyResearchAgent`
- `data` — registry-backed ReAct loop with data tools
- `mcp` — registry-backed ReAct loop over tools auto-discovered from configured MCP servers (`runtime_config['mcp_servers']` or `request.metadata['mcp_servers']`)

## Smoke checks run

```text
python -m py_compile runtime_contracts.py event_bus.py registry_bootstrap.py tool_runtime.py memory_hub.py agent_adapters.py strategy_router.py agent_runtime.py blackboard_runtime.py evaluation_runtime.py resilience.py optimization_runtime.py protocol_runtime.py lifecycle_runtime.py capstone_runtime.py parser.py test_resilience.py
```

Focused deterministic tests run:

```text
python -m pytest test_protocol_runtime.py test_resilience.py test_runtime_integration.py test_optimization_runtime.py test_evaluation_runtime.py test_approval_runtime.py test_lifecycle_runtime.py -q
# 49 passed
```

Additional manual checks:

- full registry builds
- calculator tool works through `ToolRuntime`
- schema FS + `data_read` works through registry
- `execute_python_secure` runs `print(2+2)` through `CodeExecutor`
- prompt-injection input is blocked by `AgentRuntime`
- `confirm_action` is approval-gated by default policy
- manual approval round-trip verified: `request_delete` → pending approval → approve → `resume_approval()`
- feedback revision path verified through `AgentRuntime.revise_with_feedback()`
- interactive trace verified through request-scoped runtime events
- notebook-28 resilience primitives verified: retry feedback, fallback sequencing, circuit-breaker trip/recovery, graceful degradation summaries, timeout enforcement, and `SafeToolExecutor`

## Live Notebook 25 harness

Run:

```text
python live_test_hitl_runtime.py
```

This validates the real LLM path for:
- manual approval pause
- human approve → runtime resume
- auto-allow mode
- feedback-driven revision
- interactive trace events

## Important compatibility note

The original files remain in place for lessons and comparisons. New notebook work should target `AgentRuntime` and `AgentResult` instead of creating another independent loop.
