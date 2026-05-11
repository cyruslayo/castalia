"""Protocol runtime for Notebook 29/30: local MCP + A2A foundations.

Implements a notebook-sized Model Context Protocol stack with:
- JSON-RPC style MCP messages
- tool schemas
- MCP server/client handshake and tool discovery
- multi-server MCP agent
- runtime bridge over ToolRuntime / ToolRegistry
- local A2A directory + agent cards
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from runtime_contracts import AgentRequest, AgentResult
from tool_definitions import ParameterSchema, ToolDefinition
from tool_registry import ToolRegistry, ToolResult
from tool_runtime import ToolRuntime


_JSON_TYPE_MAP = {
    "string": "string",
    "number": "number",
    "integer": "integer",
    "boolean": "boolean",
    "array": "list",
    "object": "dict",
}


# ============================================================
# MCP message format
# ============================================================


def _new_id(prefix: str = "mcp") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class MCPRequest:
    """A request from client to server."""

    method: str
    params: Dict[str, Any]
    id: str = field(default_factory=lambda: _new_id("req"))

    def to_dict(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }


@dataclass
class MCPResponse:
    """A successful server response."""

    result: Any
    id: str = ""

    def to_dict(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "result": self.result,
            "id": self.id,
        }


@dataclass
class MCPError:
    """An error server response."""

    code: int
    message: str
    id: str = ""
    data: Any = None

    def to_dict(self) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "error": {"code": self.code, "message": self.message},
            "id": self.id,
        }
        if self.data is not None:
            payload["error"]["data"] = self.data
        return payload


@dataclass
class MCPNotification:
    """A one-way message with no response expected."""

    method: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
        }


class ErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    TOOL_NOT_FOUND = -32001
    TOOL_EXECUTION_ERROR = -32002


# ============================================================
# MCP tool schema
# ============================================================


@dataclass
class ToolSchema:
    """A single MCP-exposed tool definition."""

    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    @classmethod
    def from_tool_definition(cls, definition: ToolDefinition) -> "ToolSchema":
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param in definition.parameters:
            entry: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                entry["default"] = param.default
            if param.enum:
                entry["enum"] = list(param.enum)
            if param.min_value is not None:
                entry["minimum"] = param.min_value
            if param.max_value is not None:
                entry["maximum"] = param.max_value
            if param.pattern:
                entry["pattern"] = param.pattern
            properties[param.name] = entry
            if param.required:
                required.append(param.name)
        return cls(
            name=definition.name,
            description=definition.description,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )


# ============================================================
# MCP server
# ============================================================


class MCPServer:
    """A simplified MCP server that registers and exposes tools."""

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, ToolSchema] = {}
        self.handlers: Dict[str, Callable[..., Any]] = {}
        self.capabilities = {"tools": {"listChanged": True}}
        self.supported_versions = ["2024-11-05"]
        self._log: List[str] = []

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        self.tools[name] = ToolSchema(name=name, description=description, input_schema=input_schema)
        self.handlers[name] = handler
        self._log.append(f"Registered tool: {name}")

    def register_schema(self, schema: ToolSchema, handler: Callable[..., Any]) -> None:
        self.tools[schema.name] = schema
        self.handlers[schema.name] = handler
        self._log.append(f"Registered tool: {schema.name}")

    def register_registry(self, registry: ToolRegistry) -> int:
        count = 0
        for name, definition in registry.definitions.items():
            schema = ToolSchema.from_tool_definition(definition)

            def _handler(_tool_name=name, **kwargs):
                result = registry.call(_tool_name, **kwargs)
                if not result.success:
                    raise RuntimeError(result.error or f"Tool '{_tool_name}' failed")
                return result.result

            self.register_schema(schema, _handler)
            count += 1
        return count

    def register_tool_runtime(self, tool_runtime: ToolRuntime) -> int:
        count = 0
        for name, definition in tool_runtime.registry.definitions.items():
            schema = ToolSchema.from_tool_definition(definition)

            def _handler(_tool_name=name, **kwargs):
                result = tool_runtime.call(_tool_name, kwargs)
                if not result.success:
                    raise RuntimeError(result.error or f"Tool '{_tool_name}' failed")
                return result.result

            self.register_schema(schema, _handler)
            count += 1
        return count

    def handle_request(self, request: MCPRequest) -> Union[MCPResponse, MCPError]:
        self._log.append(f"Received: {request.method} (id={request.id})")
        if request.method == "initialize":
            return self._handle_initialize(request)
        if request.method == "tools/list":
            return self._handle_list_tools(request)
        if request.method == "tools/call":
            return self._handle_call_tool(request)
        return MCPError(
            code=ErrorCode.METHOD_NOT_FOUND,
            message=f"Unknown method: {request.method}",
            id=request.id,
        )

    def handle_dict(self, request: dict) -> dict:
        normalized = MCPRequest(
            method=request.get("method", ""),
            params=request.get("params", {}) or {},
            id=request.get("id") or _new_id("req"),
        )
        response = self.handle_request(normalized)
        return response.to_dict()

    def _handle_initialize(self, request: MCPRequest) -> MCPResponse:
        client_version = request.params.get("protocolVersion", self.supported_versions[0])
        negotiated_version = client_version if client_version in self.supported_versions else self.supported_versions[0]
        payload = {
            "protocolVersion": negotiated_version,
            "serverInfo": {"name": self.name, "version": self.version},
            "capabilities": self.capabilities,
        }
        if client_version not in self.supported_versions:
            payload["warning"] = (
                f"Client requested {client_version}, server supports {self.supported_versions}"
            )
        return MCPResponse(result=payload, id=request.id)

    def _handle_list_tools(self, request: MCPRequest) -> MCPResponse:
        return MCPResponse(
            result={"tools": [schema.to_dict() for schema in self.tools.values()]},
            id=request.id,
        )

    def _handle_call_tool(self, request: MCPRequest) -> Union[MCPResponse, MCPError]:
        tool_name = request.params.get("name", "")
        arguments = request.params.get("arguments", {}) or {}
        if tool_name not in self.tools:
            return MCPError(
                code=ErrorCode.TOOL_NOT_FOUND,
                message=f"Tool '{tool_name}' not found. Available: {sorted(self.tools.keys())}",
                id=request.id,
            )
        try:
            result = self.handlers[tool_name](**arguments)
            return MCPResponse(
                result={"content": [{"type": "text", "text": str(result)}]},
                id=request.id,
            )
        except TypeError as e:
            return MCPError(
                code=ErrorCode.INVALID_PARAMS,
                message=f"Invalid parameters for '{tool_name}': {e}",
                id=request.id,
            )
        except Exception as e:
            return MCPError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Tool '{tool_name}' failed: {e}",
                id=request.id,
            )

    def get_log(self) -> List[str]:
        return list(self._log)


class CapableServer(MCPServer):
    """Extended MCP server with richer capability negotiation."""

    def __init__(self, name: str, version: str = "1.0"):
        super().__init__(name, version)
        self.capabilities = {
            "tools": {"listChanged": True},
            "resources": {"subscribe": False, "listChanged": False},
            "prompts": {"listChanged": False},
            "logging": {},
        }

    def _handle_initialize(self, request: MCPRequest) -> MCPResponse:
        client_version = request.params.get("protocolVersion", self.supported_versions[0])
        client_caps = request.params.get("capabilities", {}) or {}
        negotiated_version = client_version if client_version in self.supported_versions else self.supported_versions[0]
        payload = {
            "protocolVersion": negotiated_version,
            "serverInfo": {"name": self.name, "version": self.version},
            "capabilities": self.capabilities,
            "negotiated": {
                "tools": True,
                "resources": "resources" in client_caps,
                "prompts": "prompts" in client_caps,
            },
        }
        if client_version not in self.supported_versions:
            payload["warning"] = (
                f"Client requested {client_version}, server supports {self.supported_versions}"
            )
        return MCPResponse(result=payload, id=request.id)


class LocalMCPServer(MCPServer):
    """Compatibility wrapper: local MCP facade over ToolRuntime."""

    def __init__(self, tool_runtime: ToolRuntime, name: str = "local-mcp-server", version: str = "1.0"):
        super().__init__(name=name, version=version)
        self.tool_runtime = tool_runtime
        self.register_tool_runtime(tool_runtime)

    def handle(self, request: dict) -> dict:
        return self.handle_dict(request)


# ============================================================
# MCP client
# ============================================================


class MCPClient:
    """A simplified MCP client that connects to servers and calls tools."""

    def __init__(self, name: str = "mcp-client"):
        self.name = name
        self.server: Optional[MCPServer] = None
        self.server_info: Dict[str, Any] = {}
        self.available_tools: List[Dict[str, Any]] = []
        self.connected = False
        self._log: List[str] = []

    def connect(self, server: MCPServer) -> bool:
        self._log.append("Connecting to server...")
        init_request = MCPRequest(
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": self.name, "version": "1.0"},
                "capabilities": {"tools": True},
            },
        )
        response = server.handle_request(init_request)
        if isinstance(response, MCPError):
            self._log.append(f"Connection failed: {response.message}")
            return False
        self.server = server
        self.server_info = response.result.get("serverInfo", {})
        self.connected = True
        self._log.append(f"Connected to: {self.server_info.get('name', 'unknown')}")
        self._discover_tools()
        return True

    def _discover_tools(self) -> None:
        if not self.connected or self.server is None:
            return
        response = self.server.handle_request(MCPRequest(method="tools/list", params={}))
        if isinstance(response, MCPResponse):
            self.available_tools = response.result.get("tools", [])
            self._log.append(f"Discovered {len(self.available_tools)} tools")

    def list_available_tools(self) -> List[Dict[str, Any]]:
        return list(self.available_tools)

    def get_tool_names(self) -> List[str]:
        return [tool["name"] for tool in self.available_tools]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self.connected or self.server is None:
            return {"error": "Not connected to any server"}
        request = MCPRequest(method="tools/call", params={"name": name, "arguments": arguments})
        self._log.append(f"Calling tool: {name}({arguments})")
        response = self.server.handle_request(request)
        if isinstance(response, MCPError):
            self._log.append(f"Tool error: {response.message}")
            return {"error": response.message, "code": response.code}
        self._log.append("Tool result received")
        return {"result": response.result}

    def get_log(self) -> List[str]:
        return list(self._log)


# ============================================================
# Example MCP servers from Notebook 29
# ============================================================


def build_calculator_server() -> MCPServer:
    server = MCPServer("calculator-server", version="1.0")

    def add(a: float, b: float) -> str:
        return f"{a} + {b} = {a + b}"

    def subtract(a: float, b: float) -> str:
        return f"{a} - {b} = {a - b}"

    def multiply(a: float, b: float) -> str:
        return f"{a} * {b} = {a * b}"

    def divide(a: float, b: float) -> str:
        if b == 0:
            raise ValueError("Division by zero is undefined")
        return f"{a} / {b} = {a / b}"

    two_numbers_schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    }

    server.register_tool("add", "Add two numbers.", two_numbers_schema, add)
    server.register_tool("subtract", "Subtract b from a.", two_numbers_schema, subtract)
    server.register_tool("multiply", "Multiply two numbers.", two_numbers_schema, multiply)
    server.register_tool("divide", "Divide a by b.", two_numbers_schema, divide)
    return server


def build_text_server() -> MCPServer:
    server = MCPServer("text-processing-server", version="1.0")

    def word_count(text: str) -> str:
        words = text.split()
        chars = len(text)
        sentences = text.count(".") + text.count("!") + text.count("?")
        return f"Words: {len(words)}, Characters: {chars}, Sentences: {max(sentences, 1)}"

    def reverse_text(text: str) -> str:
        return text[::-1]

    def extract_keywords(text: str, top_n: int = 5) -> str:
        words = text.lower().split()
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "it", "in", "on", "at", "to",
            "for", "of", "and", "or", "but", "with", "that", "this", "from", "by", "as",
            "be", "has", "have",
        }
        filtered = [w.strip(".,!?;:") for w in words if w.strip(".,!?;:") not in stop_words]
        freq = defaultdict(int)
        for word in filtered:
            if len(word) > 2:
                freq[word] += 1
        top = sorted(freq.items(), key=lambda item: item[1], reverse=True)[:top_n]
        return "Keywords: " + ", ".join(f"{word}({count})" for word, count in top)

    def text_stats(text: str) -> str:
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / max(len(words), 1)
        return (
            f"Avg word length: {avg_word_len:.1f} chars, "
            f"Unique words: {unique_words}/{len(words)}, "
            f"Lexical diversity: {lexical_diversity:.2%}"
        )

    text_schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Input text to process"}},
        "required": ["text"],
    }
    keyword_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Input text"},
            "top_n": {"type": "integer", "description": "Number of keywords to extract", "default": 5},
        },
        "required": ["text"],
    }

    server.register_tool("word_count", "Count words, characters, and sentences in text.", text_schema, word_count)
    server.register_tool("reverse_text", "Reverse the input text.", text_schema, reverse_text)
    server.register_tool("extract_keywords", "Extract top keywords from text.", keyword_schema, extract_keywords)
    server.register_tool("text_stats", "Compute text statistics.", text_schema, text_stats)
    return server


def build_conversion_server() -> MCPServer:
    server = MCPServer("conversion-server", version="1.0")

    def celsius_to_fahrenheit(celsius: float) -> str:
        f = celsius * 9 / 5 + 32
        return f"{celsius} C = {f:.1f} F"

    def kg_to_pounds(kg: float) -> str:
        lbs = kg * 2.20462
        return f"{kg} kg = {lbs:.2f} lbs"

    def km_to_miles(km: float) -> str:
        miles = km * 0.621371
        return f"{km} km = {miles:.2f} miles"

    server.register_tool(
        "celsius_to_fahrenheit",
        "Convert temperature from Celsius to Fahrenheit.",
        {"type": "object", "properties": {"celsius": {"type": "number"}}, "required": ["celsius"]},
        celsius_to_fahrenheit,
    )
    server.register_tool(
        "kg_to_pounds",
        "Convert mass from kilograms to pounds.",
        {"type": "object", "properties": {"kg": {"type": "number"}}, "required": ["kg"]},
        kg_to_pounds,
    )
    server.register_tool(
        "km_to_miles",
        "Convert distance from kilometers to miles.",
        {"type": "object", "properties": {"km": {"type": "number"}}, "required": ["km"]},
        km_to_miles,
    )
    return server


# ============================================================
# MCP-aware multi-server agent
# ============================================================


class MCPAgent:
    """An agent that discovers tools from MCP servers and uses them."""

    def __init__(self, name: str = "mcp-agent"):
        self.name = name
        self.clients: Dict[str, MCPClient] = {}
        self.tool_catalog: Dict[str, Dict[str, Any]] = {}
        self._log: List[str] = []

    def connect_to_server(self, server: MCPServer) -> int:
        client = MCPClient(f"{self.name}-client-{server.name}")
        if not client.connect(server):
            self._log.append(f"Failed to connect to {server.name}")
            return 0
        self.clients[server.name] = client
        new_tools = 0
        for tool in client.list_available_tools():
            self.tool_catalog[tool["name"]] = {"server": server.name, "schema": tool}
            new_tools += 1
        self._log.append(f"Connected to {server.name}: discovered {new_tools} tools")
        return new_tools

    def get_tool_descriptions(self) -> str:
        if not self.tool_catalog:
            return "No tools available."
        lines = []
        for name, info in sorted(self.tool_catalog.items()):
            schema = info["schema"]
            desc = schema.get("description", "No description")
            params = schema.get("inputSchema", {}).get("properties", {})
            param_str = ", ".join(f"{key}: {value.get('type', 'any')}" for key, value in params.items())
            lines.append(f"- {name}({param_str}): {desc}")
        return "\n".join(lines)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name not in self.tool_catalog:
            return f"Error: Tool '{tool_name}' not found in catalog."
        server_name = self.tool_catalog[tool_name]["server"]
        result = self.clients[server_name].call_tool(tool_name, arguments)
        if "error" in result:
            return f"Error: {result['error']}"
        content = result["result"].get("content", [])
        if content:
            return content[0].get("text", str(content))
        return str(result["result"])

    def get_log(self) -> List[str]:
        return list(self._log)


# ============================================================
# Runtime bridge helpers
# ============================================================


def build_runtime_mcp_server(tool_runtime: ToolRuntime, name: str = "runtime-mcp-server", version: str = "1.0") -> LocalMCPServer:
    return LocalMCPServer(tool_runtime=tool_runtime, name=name, version=version)


# ============================================================
# MCP discovery -> ToolRegistry bridge
# ============================================================


def _parameter_from_mcp_property(name: str, schema: Dict[str, Any], required_names: List[str]) -> ParameterSchema:
    return ParameterSchema(
        name=name,
        type=_JSON_TYPE_MAP.get(schema.get("type", "string"), "string"),
        description=schema.get("description", f"Parameter '{name}'"),
        required=name in required_names,
        default=schema.get("default"),
        enum=schema.get("enum"),
        min_value=schema.get("minimum"),
        max_value=schema.get("maximum"),
        pattern=schema.get("pattern"),
    )


def tool_definition_from_mcp_tool(tool_schema: Dict[str, Any], *, alias: Optional[str] = None, category: str = "mcp") -> ToolDefinition:
    input_schema = tool_schema.get("inputSchema", {}) or {}
    properties = input_schema.get("properties", {}) or {}
    required = input_schema.get("required", []) or []
    parameters = [_parameter_from_mcp_property(name, prop, required) for name, prop in properties.items()]
    tool_name = alias or tool_schema["name"]
    description = tool_schema.get("description", "MCP-discovered tool")
    if alias and alias != tool_schema["name"]:
        description = f"{description} [MCP alias for {tool_schema['name']}]"
    return ToolDefinition(
        name=tool_name,
        description=description,
        parameters=parameters,
        return_type="string",
        return_description="MCP tool result content",
        category=category,
        version="mcp-1.0",
    )


class MCPRegistryBundle:
    """Discovered MCP tools exposed as a local ToolRegistry."""

    def __init__(self, registry: ToolRegistry, clients: Dict[str, MCPClient], tool_index: Dict[str, Dict[str, Any]]):
        self.registry = registry
        self.clients = clients
        self.tool_index = tool_index

    def stats(self) -> dict:
        return {
            "servers": sorted(self.clients.keys()),
            "tools": sorted(self.tool_index.keys()),
            "count": len(self.tool_index),
        }


def build_registry_from_mcp_servers(servers: List[MCPServer]) -> MCPRegistryBundle:
    registry = ToolRegistry()
    clients: Dict[str, MCPClient] = {}
    tool_index: Dict[str, Dict[str, Any]] = {}
    seen_names: Dict[str, str] = {}

    for server in servers or []:
        client = MCPClient(f"registry-bridge-{server.name}")
        if not client.connect(server):
            raise RuntimeError(f"Failed to connect to MCP server '{server.name}'")
        clients[server.name] = client
        for tool_schema in client.list_available_tools():
            original_name = tool_schema["name"]
            alias = original_name
            if alias in seen_names:
                alias = f"{server.name}__{original_name}"
                if seen_names[original_name] == original_name:
                    previous = tool_index[original_name]
                    previous_alias = f"{previous['server']}__{previous['original_name']}"
                    if previous_alias not in registry.tools:
                        prev_schema = tool_definition_from_mcp_tool(previous["schema"], alias=previous_alias)
                        prev_func = registry.tools.pop(original_name)
                        registry.definitions.pop(original_name)
                        registry.register(prev_schema, prev_func, warn_on_overwrite=False)
                        tool_index[previous_alias] = {**previous, "alias": previous_alias}
                        del tool_index[original_name]
            seen_names[original_name] = alias

            def _proxy(_server_name=server.name, _tool_name=original_name, **kwargs):
                result = clients[_server_name].call_tool(_tool_name, kwargs)
                if "error" in result:
                    return {"success": False, "error": result["error"]}
                content = result["result"].get("content", [])
                if content:
                    return content[0].get("text", str(content))
                return str(result["result"])

            registry.register(
                tool_definition_from_mcp_tool(tool_schema, alias=alias),
                _proxy,
                warn_on_overwrite=False,
            )
            tool_index[alias] = {
                "server": server.name,
                "original_name": original_name,
                "alias": alias,
                "schema": tool_schema,
            }

    return MCPRegistryBundle(registry=registry, clients=clients, tool_index=tool_index)


# ============================================================
# Local A2A scaffolds
# ============================================================


def _tokenize_capability_text(text: str) -> set[str]:
    return {part.strip(" ,.:;!?-_/").lower() for part in text.split() if part.strip(" ,.:;!?-_/\n\t")}


@dataclass
class AgentCard:
    name: str
    description: str
    capabilities: List[str]
    endpoint: str = "local"
    version: str = "1.0"
    auth: str = "none"
    max_concurrent_tasks: int = 5
    supported_input_types: List[str] = field(default_factory=lambda: ["text"])
    supported_output_types: List[str] = field(default_factory=lambda: ["text"])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        payload = asdict(self)
        payload["authRequired"] = self.auth != "none"
        return payload

    def matches_capability(self, query: str) -> float:
        query_terms = _tokenize_capability_text(query)
        if not query_terms:
            return 0.0
        card_terms = _tokenize_capability_text(self.description)
        for capability in self.capabilities:
            card_terms.update(_tokenize_capability_text(capability))
        overlap = query_terms & card_terms
        return len(overlap) / len(query_terms)


class TaskStatus(Enum):
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"


@dataclass
class TaskRequest:
    task_id: str
    sender: str
    description: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "taskId": self.task_id,
            "sender": self.sender,
            "description": self.description,
            "inputData": self.input_data,
            "priority": self.priority,
            "context": self.context,
            "metadata": self.metadata,
        }


@dataclass
class TaskResponse:
    task_id: str
    status: TaskStatus
    result: Any = None
    error_message: str = ""
    delegated_to: str = ""
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "taskId": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "processingTime": self.processing_time,
            "metadata": self.metadata,
        }
        if self.error_message:
            payload["errorMessage"] = self.error_message
        if self.delegated_to:
            payload["delegatedTo"] = self.delegated_to
        return payload


@dataclass
class TaskRecord:
    request: TaskRequest
    response: Optional[TaskResponse] = None
    status: TaskStatus = TaskStatus.SUBMITTED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update_status(self, status: TaskStatus) -> None:
        self.status = status
        self.updated_at = time.time()


class AgentDirectory:
    def __init__(self):
        self.cards: Dict[str, AgentCard] = {}
        self.runtimes: Dict[str, Any] = {}
        self._task_log: List[TaskRecord] = []

    def register(self, card: AgentCard, runtime=None):
        self.cards[card.name] = card
        if runtime is not None:
            self.runtimes[card.name] = runtime

    def unregister(self, name: str) -> bool:
        removed = False
        if name in self.cards:
            del self.cards[name]
            removed = True
        if name in self.runtimes:
            del self.runtimes[name]
            removed = True
        return removed

    def get(self, name: str) -> Optional[AgentCard]:
        return self.cards.get(name)

    def list_cards(self) -> List[dict]:
        return [card.to_dict() for card in self.cards.values()]

    def discover(self, capability_query: str, min_score: float = 0.1) -> List[Tuple[AgentCard, float]]:
        matches: List[Tuple[AgentCard, float]] = []
        for card in self.cards.values():
            score = card.matches_capability(capability_query)
            if score >= min_score:
                matches.append((card, score))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches

    def dispatch(self, agent_name: str, task: str) -> dict:
        if agent_name not in self.runtimes:
            return {"success": False, "error": f"No local runtime for agent '{agent_name}'"}
        result = self.runtimes[agent_name].run(AgentRequest(task=task))
        return result.to_dict()

    def dispatch_task(self, agent_name: str, request: TaskRequest) -> TaskResponse:
        runtime = self.runtimes.get(agent_name)
        if runtime is None:
            response = TaskResponse(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error_message=f"No local runtime for agent '{agent_name}'",
            )
            self._task_log.append(TaskRecord(request=request, response=response, status=TaskStatus.FAILED))
            return response

        record = TaskRecord(request=request)
        self._task_log.append(record)
        record.update_status(TaskStatus.IN_PROGRESS)
        started = time.time()
        try:
            result = runtime.run(AgentRequest(
                task=request.description,
                user_id=request.sender,
                strategy=request.metadata.get("strategy", "react"),
                metadata={**request.metadata, "a2a": True, "input_data": request.input_data, "context": request.context},
            ))
            response = TaskResponse(
                task_id=request.task_id,
                status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                result=result.to_dict(),
                error_message="; ".join(result.errors) if result.errors else "",
                processing_time=time.time() - started,
                metadata={"agent": agent_name, "strategy_used": result.strategy_used},
            )
            record.response = response
            record.update_status(response.status)
            return response
        except Exception as exc:
            response = TaskResponse(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error_message=f"{type(exc).__name__}: {exc}",
                processing_time=time.time() - started,
                metadata={"agent": agent_name},
            )
            record.response = response
            record.update_status(TaskStatus.FAILED)
            return response

    def task_log(self) -> List[TaskRecord]:
        return list(self._task_log)


class A2AAgent:
    def __init__(self, card: AgentCard, directory: AgentDirectory, runtime):
        self.card = card
        self.directory = directory
        self.runtime = runtime
        self.directory.register(card, runtime=runtime)
        self.task_inbox: List[TaskRecord] = []
        self.task_outbox: List[TaskRecord] = []

    def discover_agents(self, capability: str, min_score: float = 0.1) -> List[Tuple[str, float]]:
        return [
            (card.name, score)
            for card, score in self.directory.discover(capability, min_score=min_score)
            if card.name != self.card.name
        ]

    def receive_task(self, request: TaskRequest) -> TaskResponse:
        response = self.directory.dispatch_task(self.card.name, request)
        status = TaskStatus(response.status.value)
        self.task_inbox.append(TaskRecord(request=request, response=response, status=status))
        return response

    def send_task(
        self,
        target_name: str,
        description: str,
        input_data: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskResponse:
        request = TaskRequest(
            task_id=_new_id("task"),
            sender=self.card.name,
            description=description,
            input_data=input_data or {},
            priority=priority,
            context=context,
            metadata=metadata or {},
        )
        response = self.directory.dispatch_task(target_name, request)
        self.task_outbox.append(TaskRecord(request=request, response=response, status=response.status))
        return response


class CoordinatorAgent(A2AAgent):
    def coordinate(self, user_task: str, plan: Optional[List[Dict[str, Any]]] = None) -> AgentResult:
        plan = plan or [
            {"capability": "research", "description": f"Research: {user_task}"},
            {"capability": "analysis", "description": f"Analyze: {user_task}"},
            {"capability": "writing", "description": f"Write a polished response: {user_task}"},
        ]
        phase_results = []
        for subtask in plan:
            candidates = self.discover_agents(subtask["capability"])
            if not candidates:
                phase_results.append({
                    "capability": subtask["capability"],
                    "success": False,
                    "error": f"No agent found for capability '{subtask['capability']}'",
                })
                continue
            target_name, score = candidates[0]
            response = self.send_task(
                target_name=target_name,
                description=subtask["description"],
                input_data=subtask.get("input_data", {}),
                context=user_task,
                metadata=subtask.get("metadata", {}),
            )
            phase_results.append({
                "capability": subtask["capability"],
                "target": target_name,
                "score": score,
                "response": response.to_dict(),
                "success": response.status == TaskStatus.COMPLETED,
            })

        success = all(item.get("success") for item in phase_results) if phase_results else False
        summary_lines = [f"A2A coordination result for: {user_task}"]
        for item in phase_results:
            if item.get("success"):
                agent_result = ((item.get("response") or {}).get("result") or {})
                summary_lines.append(f"[{item['capability']}] {item['target']}: {agent_result.get('answer', '')}")
            else:
                summary_lines.append(f"[{item['capability']}] FAILED: {item.get('error') or (item.get('response') or {}).get('errorMessage', 'unknown error')}")
        return AgentResult(
            answer="\n".join(summary_lines),
            success=success,
            strategy_used="a2a_coordinator",
            artifacts={"phase_results": phase_results},
            metadata={"delegated_agents": [item.get("target") for item in phase_results if item.get("target")]},
        ).finish()
