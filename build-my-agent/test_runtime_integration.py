from agent_runtime import AgentRuntime
from optimization_runtime import RuntimeOptimizer
from protocol_runtime import build_calculator_server
from runtime_contracts import AgentContext, AgentRequest, AgentResult
from evaluation_runtime import GoldenTask


class FakeAdapter:
    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        tool = "calculator" if "347 * 23" in request.task or request.task == "math" else "search_kb"
        answer = "7981" if tool == "calculator" else "Paris"
        return AgentResult(
            answer=answer,
            success=True,
            strategy_used="react",
            steps=[{"step": 1}],
            tool_calls=[{"tool": tool, "params": {}, "success": True}],
            metadata={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
            request_id=request.request_id,
        ).finish()


def test_registry_react_adapter_receives_optimizer_and_surfaces_token_metadata(monkeypatch):
    import agent_tool_integration
    from agent_adapters import RegistryReactAdapter

    captured = {}

    class FakeAgent:
        def __init__(self, tool_backend, max_steps=8, event_bus=None, request_id=None, actor="agent", optimizer=None):
            captured["optimizer"] = optimizer

        def run(self, query: str, verbose: bool = True):
            return {
                "answer": "7981",
                "steps": [{"step": 1}],
                "tool_calls": 1,
                "total_time": 0.01,
                "input_tokens": 10,
                "output_tokens": 3,
                "total_tokens": 13,
                "optimization": {"cache": {"hits": 1}},
            }

    monkeypatch.setattr(agent_tool_integration, "AdvancedToolAgent", FakeAgent)

    optimizer = RuntimeOptimizer()
    context = AgentContext(
        request=AgentRequest(task="What is 347 * 23?"),
        tool_runtime=type("TR", (), {"recent_calls": lambda self, limit=20: []})(),
        runtime_config={},
        optimizer=optimizer,
    )
    result = RegistryReactAdapter().run(AgentRequest(task="What is 347 * 23?"), context)

    assert captured["optimizer"] is optimizer
    assert result.metadata["input_tokens"] == 10
    assert result.metadata["output_tokens"] == 3
    assert result.metadata["total_tokens"] == 13
    assert result.metadata["optimization"]["cache"]["hits"] == 1


def test_agent_runtime_stats_include_optimization_block():
    runtime = AgentRuntime(adapters={"react": FakeAdapter(), "auto": FakeAdapter()}, runtime_config={"max_steps": 1, "verbose": False})
    stats = runtime.stats()
    assert "optimization" in stats
    assert "budget" in stats["optimization"]
    assert "cache" in stats["optimization"]


def test_agent_runtime_evaluate_dataset_uses_notebook26_harness():
    runtime = AgentRuntime(adapters={"react": FakeAdapter(), "auto": FakeAdapter()}, runtime_config={"max_steps": 1, "verbose": False})
    dataset = [
        GoldenTask("math-001", "math", expected_answer="7981", expected_tools=["calculator"], category="math"),
        GoldenTask("fact-001", "fact", expected_answer="Paris", expected_tools=["search_kb"], category="factual"),
    ]

    evaluator = runtime.evaluate_dataset(dataset=dataset, use_judge=False, name="RuntimeEval")
    report = evaluator.aggregate_report()

    assert report["agent_name"] == "RuntimeEval"
    assert report["total_tasks"] == 2
    assert report["passed"] == 2
    assert report["pass_rate"] == 1.0


def test_mcp_adapter_discovers_remote_tools_and_exposes_metadata(monkeypatch):
    import agent_tool_integration

    captured = {}

    class FakeAgent:
        def __init__(self, tool_backend, max_steps=8, event_bus=None, request_id=None, actor="agent", optimizer=None):
            captured["tool_names"] = sorted(tool_backend.registry.tools.keys())

        def run(self, query: str, verbose: bool = True):
            return {
                "answer": "7981",
                "steps": [{"step": 1}],
                "tool_calls": 1,
                "total_time": 0.01,
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
                "optimization": {},
            }

    monkeypatch.setattr(agent_tool_integration, "AdvancedToolAgent", FakeAgent)

    runtime = AgentRuntime(
        runtime_config={"max_steps": 2, "verbose": False, "mcp_servers": [build_calculator_server()]},
    )
    result = runtime.run("use mcp tools", strategy="mcp")

    assert result.success is True
    assert captured["tool_names"] == ["add", "divide", "multiply", "subtract"]
    assert result.metadata["mcp_servers"] == ["calculator-server"]
    assert "multiply" in result.metadata["mcp_tools"]
    events = [evt["event_type"] for evt in runtime.interactive_trace(result.request_id)]
    assert "mcp.discovery.completed" in events


def test_mcp_adapter_requires_servers_configuration():
    runtime = AgentRuntime(runtime_config={"max_steps": 1, "verbose": False})
    result = runtime.run("use mcp tools", strategy="mcp")
    assert result.success is False
    assert "No MCP servers configured" in result.answer
