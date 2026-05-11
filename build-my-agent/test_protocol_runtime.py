from agent_runtime import AgentRuntime
from protocol_runtime import (
    A2AAgent,
    AgentCard,
    AgentDirectory,
    CapableServer,
    CoordinatorAgent,
    ErrorCode,
    MCPAgent,
    MCPClient,
    MCPError,
    MCPRequest,
    MCPServer,
    TaskRequest,
    TaskStatus,
    ToolSchema,
    build_calculator_server,
    build_runtime_mcp_server,
)
from runtime_contracts import AgentContext, AgentRequest, AgentResult
from tool_definitions import ParameterSchema, ToolDefinition
from tool_registry import ToolRegistry
from tool_runtime import ToolRuntime


class FakeAdapter:
    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        return AgentResult(
            answer=f"done: {request.task}",
            success=True,
            strategy_used="react",
            request_id=request.request_id,
        ).finish()


class FailingResearchAdapter:
    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        return AgentResult(
            answer="research adapter should not have been selected",
            success=False,
            strategy_used="research",
            errors=["wrong adapter selected"],
            request_id=request.request_id,
        ).finish()



def test_mcp_request_and_tool_schema_serialization():
    req = MCPRequest(method="tools/list", params={})
    payload = req.to_dict()
    assert payload["jsonrpc"] == "2.0"
    assert payload["method"] == "tools/list"
    assert "id" in payload

    definition = ToolDefinition(
        name="echo",
        description="Echo a message.",
        parameters=[ParameterSchema("msg", "string", "Message to echo")],
        return_type="string",
        return_description="Echoed message",
    )
    schema = ToolSchema.from_tool_definition(definition).to_dict()
    assert schema["name"] == "echo"
    assert schema["inputSchema"]["properties"]["msg"]["type"] == "string"



def test_calculator_server_client_flow_and_errors():
    server = build_calculator_server()
    client = MCPClient("calc-user")

    assert client.connect(server) is True
    assert set(client.get_tool_names()) == {"add", "subtract", "multiply", "divide"}

    result = client.call_tool("add", {"a": 2, "b": 3})
    assert result["result"]["content"][0]["text"] == "2 + 3 = 5"

    error = client.call_tool("divide", {"a": 10, "b": 0})
    assert error["code"] == ErrorCode.TOOL_EXECUTION_ERROR
    assert "Division by zero" in error["error"]

    missing = server.handle_request(MCPRequest(method="tools/call", params={"name": "sqrt", "arguments": {}}))
    assert isinstance(missing, MCPError)
    assert missing.code == ErrorCode.TOOL_NOT_FOUND



def test_mcp_agent_discovers_tools_across_multiple_servers():
    calc = build_calculator_server()
    text = MCPServer("text")
    text.register_tool(
        "echo",
        "Echo input.",
        {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
        lambda msg: msg,
    )

    agent = MCPAgent("multi")
    assert agent.connect_to_server(calc) == 4
    assert agent.connect_to_server(text) == 1
    assert "multiply" in agent.tool_catalog
    assert "echo" in agent.tool_catalog
    assert agent.execute_tool("multiply", {"a": 6, "b": 7}) == "6 * 7 = 42"
    assert agent.execute_tool("echo", {"msg": "hello"}) == "hello"



def test_capable_server_negotiates_version_and_capabilities():
    server = CapableServer("capable")
    response = server.handle_request(
        MCPRequest(
            method="initialize",
            params={
                "protocolVersion": "2025-01-01",
                "capabilities": {"tools": True, "resources": True},
            },
        )
    )
    assert response.result["protocolVersion"] == "2024-11-05"
    assert "warning" in response.result
    assert response.result["negotiated"]["resources"] is True



def test_runtime_mcp_server_bridges_tool_runtime():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo a message.",
            parameters=[ParameterSchema("msg", "string", "Message")],
            return_type="string",
            return_description="Echoed message",
        ),
        lambda msg: msg,
    )
    tool_runtime = ToolRuntime(registry)
    server = build_runtime_mcp_server(tool_runtime, name="runtime-bridge")
    client = MCPClient("bridge-client")

    assert client.connect(server) is True
    assert client.get_tool_names() == ["echo"]
    result = client.call_tool("echo", {"msg": "through runtime"})
    assert result["result"]["content"][0]["text"] == "through runtime"



def test_agent_directory_dispatches_to_local_runtime():
    runtime = AgentRuntime(
        adapters={"react": FakeAdapter(), "auto": FakeAdapter()},
        runtime_config={"max_steps": 1, "verbose": False},
    )
    directory = AgentDirectory()
    directory.register(AgentCard(name="helper", description="Test helper", capabilities=["chat"]), runtime=runtime)

    result = directory.dispatch("helper", "say hi")
    assert result["success"] is True
    assert result["answer"] == "done: say hi"


def test_agent_card_discovery_and_task_dispatch():
    runtime = AgentRuntime(
        adapters={"react": FakeAdapter(), "auto": FakeAdapter()},
        runtime_config={"max_steps": 1, "verbose": False},
    )
    directory = AgentDirectory()
    helper = AgentCard(
        name="ResearchHelper",
        description="Research and fact finding helper",
        capabilities=["research", "search", "fact finding"],
    )
    directory.register(helper, runtime=runtime)

    matches = directory.discover("research facts")
    assert matches[0][0].name == "ResearchHelper"
    assert matches[0][1] > 0

    response = directory.dispatch_task(
        "ResearchHelper",
        TaskRequest(task_id="task-1", sender="user", description="say hi"),
    )
    assert response.status == TaskStatus.COMPLETED
    assert response.result["answer"] == "done: say hi"
    assert directory.task_log()[-1].status == TaskStatus.COMPLETED


def test_a2a_agent_send_task_and_coordinator_flow():
    runtime = AgentRuntime(
        adapters={"react": FakeAdapter(), "auto": FakeAdapter(), "research": FailingResearchAdapter()},
        runtime_config={"max_steps": 1, "verbose": False},
    )
    directory = AgentDirectory()
    research = A2AAgent(
        AgentCard(name="ResearchAgent", description="Research specialist", capabilities=["research", "search"]),
        directory,
        runtime,
    )
    analysis = A2AAgent(
        AgentCard(name="AnalysisAgent", description="Analysis specialist", capabilities=["analysis", "evaluation"]),
        directory,
        runtime,
    )
    writer = A2AAgent(
        AgentCard(name="WritingAgent", description="Writing specialist", capabilities=["writing", "summary"]),
        directory,
        runtime,
    )
    coordinator = CoordinatorAgent(
        AgentCard(name="CoordinatorAgent", description="Delegates to specialists", capabilities=["coordination"]),
        directory,
        runtime,
    )

    direct = coordinator.send_task("ResearchAgent", "research and collect findings")
    assert direct.status == TaskStatus.COMPLETED
    assert direct.result["answer"] == "done: research and collect findings"
    assert direct.result["strategy_used"] == "react"
    assert coordinator.task_outbox[-1].status == TaskStatus.COMPLETED

    result = coordinator.coordinate(
        "remote work",
        plan=[
            {"capability": "research", "description": "collect findings"},
            {"capability": "analysis", "description": "compare tradeoffs"},
            {"capability": "writing", "description": "write summary"},
        ],
    )
    assert result.success is True
    assert "ResearchAgent" in result.answer
    assert "AnalysisAgent" in result.answer
    assert "WritingAgent" in result.answer
