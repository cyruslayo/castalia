"""
Advanced Tool Registry — Production tool dispatch, validation, and testing.

Implements:
  - ToolResult: standardized output with timing and error tracking
  - ToolRegistry: validation, dispatch, discovery, call history, stats
  - validate_with_helpful_errors: LLM-friendly error messages
  - ToolTestHarness: automated testing (happy path, edge, error cases)
  - Stateful, Confirmation, and Composed tool patterns

Integrates with tool_definitions.py (ParameterSchema, ToolDefinition)
and tools.py (existing tool implementations).
"""

import time
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from collections import defaultdict

from tool_definitions import ParameterSchema, ToolDefinition


# ============================================================================
# ToolResult — Standardized output from any tool call
# ============================================================================

class ToolResult:
    """Standardized result from any tool call.

    Replaces string-only returns with structured data:
      - success: bool — did the tool execute without error?
      - result: Any — the tool's output (if successful)
      - error: str — human-readable error message (if failed)
      - tool_name: str — which tool was called
      - execution_time: float — seconds taken

    The to_dict() method produces JSON-friendly output for the LLM.
    """

    def __init__(self, success: bool, result: Any = None, error: str = None,
                 tool_name: str = "", execution_time: float = 0.0):
        self.success = success
        self.result = result
        self.error = error
        self.tool_name = tool_name
        self.execution_time = execution_time

    def to_dict(self) -> dict:
        """Convert to a dict suitable for JSON serialization and LLM consumption."""
        d = {
            "success": self.success,
            "tool": self.tool_name,
            "execution_time_ms": round(self.execution_time * 1000, 1),
        }
        if self.result is not None:
            d["result"] = self.result
        if not self.success:
            d["error"] = self.error
        return d

    def __repr__(self):
        if self.success:
            return f"ToolResult(success=True, result={self.result!r})"
        return f"ToolResult(success=False, error={self.error!r})"


# ============================================================================
# LLM-Friendly Validation — Error messages as corrective prompts
# ============================================================================

def validate_with_helpful_errors(definition: ToolDefinition, kwargs: dict) -> Optional[str]:
    """Validate inputs against a ToolDefinition's parameter schema.

    Returns None if all validations pass, or a formatted error string
    that tells the LLM exactly what went wrong AND how to fix it.

    Key principle: error messages are corrective prompts, not raw exceptions.
    Each error contains:
      1. What went wrong
      2. What was expected
      3. How to fix it (with examples when available)
    """
    errors = []

    # Check for unknown parameters
    known_names = {p.name for p in definition.parameters}
    unknown = set(kwargs.keys()) - known_names
    if unknown:
        errors.append(
            f"Unknown parameter(s): {unknown}. "
            f"Valid parameters for '{definition.name}': {sorted(known_names)}"
        )

    for param in definition.parameters:
        # Missing required parameter
        if param.required and param.name not in kwargs:
            example = ""
            if definition.examples:
                ex_input = definition.examples[0].get("input", {})
                if param.name in ex_input:
                    example = f" Example: {param.name}={ex_input[param.name]!r}"
            errors.append(
                f"Missing required parameter '{param.name}' ({param.type}): "
                f"{param.description}.{example}"
            )
            continue

        if param.name not in kwargs:
            continue

        value = kwargs[param.name]

        # Type check
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "list": list,
            "dict": dict,
        }
        expected = type_map.get(param.type)
        if expected and not isinstance(value, expected):
            fix_hint = ""
            if param.type == "integer" and isinstance(value, str):
                try:
                    fix_hint = f" Try passing {param.name}={int(value)} without quotes."
                except ValueError:
                    fix_hint = f" Try passing an integer for {param.name}."
            elif param.type == "number" and isinstance(value, str):
                try:
                    fix_hint = f" Try passing {param.name}={float(value)} without quotes."
                except ValueError:
                    fix_hint = f" Try passing a number for {param.name}."
            elif param.type == "string" and isinstance(value, (int, float)):
                fix_hint = f' Try passing {param.name}="{value}".'
            elif param.type == "boolean" and isinstance(value, str):
                fix_hint = f" Try passing {param.name}={value.lower() == 'true'} (boolean, not string)."
            elif param.type == "list" and isinstance(value, str):
                fix_hint = f" Try passing {param.name}={value} as a JSON array, not a string."

            errors.append(
                f"Parameter '{param.name}': expected {param.type}, got "
                f"{type(value).__name__} ({value!r}).{fix_hint}"
            )

        # Enum check
        if param.enum and value not in param.enum:
            errors.append(
                f"Parameter '{param.name}': value '{value}' is not allowed. "
                f"Choose one of: {param.enum}"
            )

        # Range check
        if isinstance(value, (int, float)):
            if param.min_value is not None and value < param.min_value:
                errors.append(
                    f"Parameter '{param.name}': {value} is below minimum {param.min_value}"
                )
            if param.max_value is not None and value > param.max_value:
                errors.append(
                    f"Parameter '{param.name}': {value} exceeds maximum {param.max_value}"
                )

        # Pattern check (regex for strings)
        if param.pattern and isinstance(value, str):
            if not re.match(param.pattern, value):
                errors.append(
                    f"Parameter '{param.name}': '{value}' doesn't match "
                    f"required pattern '{param.pattern}'"
                )

    if errors:
        return "TOOL CALL ERROR for '" + definition.name + "':\n" + "\n".join(
            f"  - {e}" for e in errors
        )
    return None


# ============================================================================
# ToolRegistry — Central hub for tool dispatch and validation
# ============================================================================

class ToolRegistry:
    """Advanced registry with validation, discovery, and error handling.

    Stores tools, validates them at registration time, dispatches calls,
    and handles errors. Think of it as a strongly-typed function dispatch table.

    Key features:
      - register(): add a tool with its ToolDefinition
      - validate_input(): check kwargs against ParameterSchema rules
      - call(): dispatch a tool call with validation, timing, and error handling
      - discover(): return formatted tool descriptions for the LLM
      - get_stats(): aggregate call history statistics
      - call_history: audit log of every tool call made

    This replaces the informal TOOLS dict in tools.py with a formal,
    testable, and extensible system.
    """

    TYPE_MAP = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "list": list,
        "dict": dict,
    }

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.definitions: Dict[str, ToolDefinition] = {}
        self.call_history: List[dict] = []

    def register(self, definition: ToolDefinition, func: Callable, warn_on_overwrite: bool = True) -> None:
        """Register a tool with its definition. Validates the definition."""
        assert definition.name, "Tool must have a name"
        assert definition.description, "Tool must have a description"
        assert callable(func), "Tool function must be callable"
        if definition.name in self.tools and warn_on_overwrite:
            print(f"  Warning: Overwriting existing tool: {definition.name}")
        self.tools[definition.name] = func
        self.definitions[definition.name] = definition

    def validate_input(self, tool_name: str, kwargs: dict) -> Optional[str]:
        """Validate inputs against the tool's parameter schema."""
        defn = self.definitions.get(tool_name)
        if not defn:
            return (
                f"Unknown tool: {tool_name}. "
                f"Available tools: {list(self.tools.keys())}"
            )

        # Use the LLM-friendly validator
        return validate_with_helpful_errors(defn, kwargs)

    def call(self, tool_name: str, **kwargs) -> ToolResult:
        """Call a tool with validation, timing, and error handling."""
        start = time.time()

        # Validate inputs
        error = self.validate_input(tool_name, kwargs)
        if error:
            result = ToolResult(False, error=error, tool_name=tool_name)
            self.call_history.append({
                "tool": tool_name,
                "args": kwargs,
                "result": result.to_dict(),
            })
            return result

        # Execute
        try:
            func = self.tools[tool_name]
            output = func(**kwargs)
            elapsed = time.time() - start

            # Detect structured error dicts returned by tools (e.g. {"success": False, "error": "..."})
            if isinstance(output, dict) and output.get("success") is False:
                result = ToolResult(
                    False,
                    error=output.get("error", "Unknown tool error"),
                    tool_name=tool_name,
                    execution_time=elapsed,
                )
            else:
                result = ToolResult(
                    True, result=output, tool_name=tool_name, execution_time=elapsed
                )
        except Exception as e:
            elapsed = time.time() - start
            result = ToolResult(
                False,
                error=f"{type(e).__name__}: {e}",
                tool_name=tool_name,
                execution_time=elapsed,
            )

        self.call_history.append({
            "tool": tool_name,
            "args": kwargs,
            "result": result.to_dict(),
        })
        return result

    def discover(self, category: str = None) -> str:
        """Return a formatted list of available tools for the LLM."""
        lines = ["Available tools:\n"]
        for name, defn in sorted(self.definitions.items()):
            if category and defn.category != category:
                continue
            lines.append(defn.to_prompt_string())
            lines.append("")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Return aggregate statistics from call history."""
        total = len(self.call_history)
        successes = sum(1 for c in self.call_history if c["result"]["success"])
        return {
            "total_calls": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": f"{successes/total:.1%}" if total else "N/A",
        }

    def per_tool_stats(self) -> dict:
        """Return per-tool breakdown of successes and failures."""
        tool_calls = defaultdict(lambda: {"success": 0, "fail": 0})
        for entry in self.call_history:
            tool = entry["tool"]
            if entry["result"]["success"]:
                tool_calls[tool]["success"] += 1
            else:
                tool_calls[tool]["fail"] += 1
        return dict(tool_calls)


# ============================================================================
# ToolTestHarness — Automated testing for tools
# ============================================================================

@dataclass
class ToolTestCase:
    """A single test case for a tool."""
    name: str
    tool_name: str
    inputs: dict
    expected_success: bool
    expected_contains: Optional[str] = None      # substring in result
    expected_error_contains: Optional[str] = None # substring in error


class ToolTestHarness:
    """Automated testing framework for tools.

    Runs happy-path, edge-case, and error-case tests against a ToolRegistry.
    Reports results in a clear table format.

    Usage:
        harness = ToolTestHarness(registry)
        harness.run_suite([
            ToolTestCase("test_name", "tool_name", {"arg": "value"}, True),
        ])
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.results: List[dict] = []

    def run_test(self, test: ToolTestCase) -> dict:
        """Run a single test case and return the result."""
        result = self.registry.call(test.tool_name, **test.inputs)
        passed = True
        reason = "OK"

        if result.success != test.expected_success:
            passed = False
            reason = f"Expected success={test.expected_success}, got {result.success}"
        elif test.expected_contains and test.expected_success:
            result_str = str(result.result)
            if test.expected_contains not in result_str:
                passed = False
                reason = f"Result missing '{test.expected_contains}'"
        elif test.expected_error_contains and not test.expected_success:
            if test.expected_error_contains not in str(result.error):
                passed = False
                reason = f"Error missing '{test.expected_error_contains}'"

        entry = {
            "test": test.name,
            "tool": test.tool_name,
            "passed": passed,
            "reason": reason,
        }
        self.results.append(entry)
        return entry

    def run_suite(self, tests: List[ToolTestCase]) -> None:
        """Run a suite of test cases and print results."""
        print(f"Running {len(tests)} tests...\n")
        print(f"{'Test':<35} {'Tool':<20} {'Result':<8} {'Reason'}")
        print("-" * 90)
        for test in tests:
            entry = self.run_test(test)
            status = "PASS" if entry["passed"] else "FAIL"
            print(
                f"{entry['test']:<35} {entry['tool']:<20} {status:<8} {entry['reason']}"
            )
        passed = sum(1 for r in self.results if r["passed"])
        print(f"\n{'=' * 90}")
        print(f"Results: {passed}/{len(self.results)} passed ({passed/len(self.results):.0%})")


# ============================================================================
# Complex Tool Types: Stateful, Confirmation, Composed
# ============================================================================

class StatefulKeyValueStore:
    """A stateful tool — an in-memory key-value store that persists across calls.

    Demonstrates how to wrap a stateful object as a set of tool functions.
    The registry calls kv.set(), kv.get(), kv.delete() as independent tools,
    but they all share the same underlying state.
    """

    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.access_log: List[dict] = []

    def set(self, key: str, value: str) -> dict:
        old = self.store.get(key)
        self.store[key] = value
        self.access_log.append({"op": "set", "key": key, "time": time.time()})
        return {"status": "updated" if old else "created", "key": key, "value": value}

    def get(self, key: str) -> dict:
        self.access_log.append({"op": "get", "key": key, "time": time.time()})
        if key in self.store:
            return {"found": True, "key": key, "value": self.store[key]}
        return {
            "found": False,
            "key": key,
            "error": f"Key '{key}' not found. Available keys: {list(self.store.keys())}",
        }

    def delete(self, key: str) -> dict:
        if key in self.store:
            del self.store[key]
            self.access_log.append({"op": "delete", "key": key, "time": time.time()})
            return {"deleted": True, "key": key}
        return {"deleted": False, "error": f"Key '{key}' not found"}

    def list_keys(self) -> dict:
        return {"keys": list(self.store.keys()), "count": len(self.store)}


class ConfirmationTool:
    """Tool that requires explicit confirmation for destructive operations.

    Demonstrates the two-phase commit pattern:
      1. request_delete() returns a confirmation token instead of deleting
      2. confirm_action() uses the token to actually perform the deletion

    This prevents accidental data loss and gives the agent a chance to
    preview the effect before committing.
    """

    def __init__(self):
        self.pending_actions: Dict[str, dict] = {}
        self.action_counter = 0

    def request_delete(self, resource_type: str, resource_id: str) -> dict:
        """Request deletion — returns a confirmation token."""
        self.action_counter += 1
        token = f"confirm_{self.action_counter}"
        self.pending_actions[token] = {
            "action": "delete",
            "resource_type": resource_type,
            "resource_id": resource_id,
            "created": time.time(),
        }
        return {
            "status": "confirmation_required",
            "message": f"To delete {resource_type} '{resource_id}', call confirm_action with token '{token}'",
            "token": token,
            "preview": f"This will permanently delete {resource_type} '{resource_id}'",
        }

    def confirm_action(self, token: str) -> dict:
        """Confirm a pending action using its token."""
        if token not in self.pending_actions:
            return {
                "success": False,
                "error": f"Invalid or expired token: {token}. "
                f"Pending tokens: {list(self.pending_actions.keys())}",
            }
        action = self.pending_actions.pop(token)
        return {
            "success": True,
            "action": action["action"],
            "resource_type": action["resource_type"],
            "resource_id": action["resource_id"],
            "message": f"Successfully deleted {action['resource_type']} '{action['resource_id']}'",
        }


def composed_store_and_verify(registry: ToolRegistry, key: str, value: str) -> dict:
    """Composed tool: store a value, then immediately read it back to verify.

    Demonstrates how to build complex workflows from simple tools.
    This calls kv_set and kv_get internally via the registry.
    """
    # Step 1: store
    set_result = registry.call("kv_set", key=key, value=value)
    if not set_result.success:
        return {"success": False, "step": "store", "error": set_result.error}

    # Step 2: verify by reading back
    get_result = registry.call("kv_get", key=key)
    if not get_result.success:
        return {"success": False, "step": "verify", "error": get_result.error}

    # Step 3: compare
    stored_value = get_result.result.get("value")
    verified = stored_value == value
    return {
        "success": verified,
        "key": key,
        "stored_value": stored_value,
        "verified": verified,
        "steps_taken": ["kv_set", "kv_get", "compare"],
    }


# ============================================================================
# Registry Builder — Wire all tools together
# ============================================================================

def build_production_registry() -> ToolRegistry:
    """Build a complete production tool registry with all tools from tools.py
    plus the new advanced tools from this module.

    This is the integration point between the old tools.py and the new
    formal registry system.
    """
    from tools import (
        run_python, read_file, write_file, search_kb, search_web
    )

    registry = ToolRegistry()

    # Import existing tool definitions from tool_definitions.py
    from tool_definitions import PYTHON_TOOL_DEF

    # Existing tools.py functions take a single `params` dict.
    # Wrap them so the registry can call them with **kwargs.
    def _wrap_params(fn):
        return lambda **kwargs: fn(kwargs)

    # Register the python tool with its formal definition
    registry.register(PYTHON_TOOL_DEF, _wrap_params(run_python))

    # Register read_file with formal definition
    registry.register(
        ToolDefinition(
            name="read_file",
            description="Read the contents of a file from the workspace.",
            parameters=[
                ParameterSchema(
                    name="filepath", type="string",
                    description="Path to the file (relative to workspace)", required=True
                ),
                ParameterSchema(
                    name="lines", type="number",
                    description="How many lines to read (default: all)", required=False
                ),
                ParameterSchema(
                    name="start_line", type="number",
                    description="1-indexed starting line (default: 1)", required=False
                ),
            ],
            return_type="string",
            return_description="The file contents, or an error message",
            category="file",
        ),
        _wrap_params(read_file),
    )

    # Register write_file with formal definition
    registry.register(
        ToolDefinition(
            name="write_file",
            description="Write content to a file in the workspace.",
            parameters=[
                ParameterSchema(
                    name="filepath", type="string",
                    description="Destination file path (relative to workspace)", required=True
                ),
                ParameterSchema(
                    name="content", type="string",
                    description="The text to write", required=True
                ),
                ParameterSchema(
                    name="append", type="boolean",
                    description="If true, append instead of overwrite", required=False,
                    default=False
                ),
            ],
            return_type="string",
            return_description="Success confirmation with character count, or an error message",
            category="file",
        ),
        _wrap_params(write_file),
    )

    # Register search_kb
    registry.register(
        ToolDefinition(
            name="search_kb",
            description="Search the internal knowledge base for facts.",
            parameters=[
                ParameterSchema(
                    name="query", type="string",
                    description="The search query. Use key terms for best results.", required=True
                ),
                ParameterSchema(
                    name="max_results", type="number",
                    description="How many results to return (default: 5, max: 10)", required=False,
                    default=5
                ),
            ],
            return_type="string",
            return_description="Formatted knowledge base results, or a message if no results found",
            category="search",
        ),
        _wrap_params(search_kb),
    )

    # Register search_web
    registry.register(
        ToolDefinition(
            name="search_web",
            description="Search the web for information and return top results.",
            parameters=[
                ParameterSchema(
                    name="query", type="string",
                    description="The search query", required=True
                ),
                ParameterSchema(
                    name="max_results", type="number",
                    description="How many results to return (default: 5, max: 10)", required=False,
                    default=5
                ),
            ],
            return_type="string",
            return_description="Formatted search results",
            category="search",
        ),
        _wrap_params(search_web),
    )

    # --- Stateful Key-Value Store ---
    kv = StatefulKeyValueStore()
    registry.register(
        ToolDefinition(
            name="kv_set",
            description="Store a key-value pair in the persistent store.",
            parameters=[
                ParameterSchema(name="key", type="string", description="The key to store"),
                ParameterSchema(name="value", type="string", description="The value to store"),
            ],
            return_type="dict",
            return_description="Status dict with 'status', 'key', 'value'",
            examples=[
                {
                    "input": {"key": "user_name", "value": "Alice"},
                    "output": {"status": "created", "key": "user_name", "value": "Alice"},
                }
            ],
            category="storage",
        ),
        kv.set,
    )

    registry.register(
        ToolDefinition(
            name="kv_get",
            description="Retrieve a value by key from the persistent store.",
            parameters=[
                ParameterSchema(name="key", type="string", description="The key to look up"),
            ],
            return_type="dict",
            return_description="Dict with 'found', 'key', 'value' or 'error'",
            category="storage",
        ),
        kv.get,
    )

    # --- Confirmation Tool ---
    confirm_tool = ConfirmationTool()
    registry.register(
        ToolDefinition(
            name="request_delete",
            description="Request deletion of a resource. Returns a confirmation token — the deletion is NOT performed until you call confirm_action with the token.",
            parameters=[
                ParameterSchema(
                    name="resource_type", type="string",
                    description="Type of resource (e.g. 'file', 'record', 'user')"
                ),
                ParameterSchema(
                    name="resource_id", type="string",
                    description="ID of the resource to delete"
                ),
            ],
            return_type="dict",
            return_description="Confirmation token and preview of the action",
            category="admin",
        ),
        confirm_tool.request_delete,
    )

    registry.register(
        ToolDefinition(
            name="confirm_action",
            description="Confirm a previously requested destructive action using the token returned by request_delete.",
            parameters=[
                ParameterSchema(
                    name="token", type="string",
                    description="The confirmation token from request_delete"
                ),
            ],
            return_type="dict",
            return_description="Result of the confirmed action",
            category="admin",
        ),
        confirm_tool.confirm_action,
    )

    # --- Composed Tool ---
    registry.register(
        ToolDefinition(
            name="store_and_verify",
            description="Store a key-value pair and immediately verify it was stored correctly. Uses kv_set and kv_get internally.",
            parameters=[
                ParameterSchema(name="key", type="string", description="The key to store"),
                ParameterSchema(name="value", type="string", description="The value to store"),
            ],
            return_type="dict",
            return_description="Dict with 'success', 'verified', 'steps_taken'",
            examples=[
                {
                    "input": {"key": "test", "value": "hello"},
                    "output": {"success": True, "verified": True},
                }
            ],
            category="storage",
        ),
        lambda key, value: composed_store_and_verify(registry, key, value),
    )

    # --- Production Tools (10 additional tools) ---
    from production_tools import register_production_tools
    register_production_tools(registry)

    return registry


# ============================================================================
# Test Suite
# ============================================================================

def run_tool_registry_tests() -> None:
    """Run a comprehensive test suite against the production registry."""
    registry = build_production_registry()
    harness = ToolTestHarness(registry)

    tests = [
        # Python tool tests
        ToolTestCase(
            "python_basic", "python", {"code": "print(2 + 2)"}, True, "4"
        ),
        ToolTestCase(
            "python_timeout", "python", {"code": "import time; time.sleep(5)", "timeout": 2},
            True, "timed out"
        ),

        # File tool tests
        ToolTestCase(
            "write_then_read", "write_file",
            {"filepath": "test_registry.txt", "content": "Hello Registry!"},
            True, "Successfully"
        ),
        ToolTestCase(
            "read_existing", "read_file",
            {"filepath": "test_registry.txt"},
            True, "Hello Registry"
        ),

        # Knowledge base tests
        ToolTestCase(
            "search_kb_basic", "search_kb", {"query": "capital France"},
            True
        ),

        # Stateful KV store tests
        ToolTestCase(
            "kv_set_basic", "kv_set", {"key": "color", "value": "blue"},
            True, "created"
        ),
        ToolTestCase(
            "kv_set_overwrite", "kv_set", {"key": "color", "value": "red"},
            True, "updated"
        ),
        ToolTestCase(
            "kv_get_existing", "kv_get", {"key": "color"},
            True, "red"
        ),
        ToolTestCase(
            "kv_get_missing", "kv_get", {"key": "nonexistent"},
            True, "not found"
        ),

        # Confirmation tool tests
        ToolTestCase(
            "confirm_request", "request_delete",
            {"resource_type": "file", "resource_id": "important.csv"},
            True, "confirmation_required"
        ),

        # Composed tool test
        ToolTestCase(
            "composed_verify", "store_and_verify",
            {"key": "test_verify", "value": "verified"},
            True, "verified"
        ),

        # Validation tests (should fail gracefully — success=False but structured error)
        ToolTestCase(
            "python_missing_code", "python", {},
            False, None, "Missing required parameter"
        ),
        ToolTestCase(
            "kv_get_missing_key", "kv_get", {},
            False, None, "Missing required parameter"
        ),
    ]

    print("=" * 90)
    print("TOOL REGISTRY TEST SUITE")
    print("=" * 90)
    harness.run_suite(tests)

    # Print registry stats
    print("\n" + "=" * 50)
    print("REGISTRY LIFETIME STATISTICS")
    print("=" * 50)
    stats = registry.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nPer-tool breakdown:")
    print(f"{'Tool':<25} {'Success':<10} {'Fail':<10} {'Total':<10} {'Rate'}")
    print("-" * 65)
    for tool, counts in sorted(registry.per_tool_stats().items()):
        total = counts["success"] + counts["fail"]
        rate = counts["success"] / total if total else 0
        print(f"{tool:<25} {counts['success']:<10} {counts['fail']:<10} {total:<10} {rate:.0%}")


if __name__ == "__main__":
    run_tool_registry_tests()
