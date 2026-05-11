"""Build the canonical full ToolRegistry for the integrated runtime.

This file is the wiring point between notebook modules.  The older tools.py
registry remains for backwards compatibility, but new runtime code should use
build_full_tool_registry() + ToolRuntime.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from tool_definitions import ParameterSchema, ToolDefinition
from tool_registry import ToolRegistry, build_production_registry


# Shared in-memory schema-aware filesystem for data-tool demos.
_SCHEMA_FS = None
_TFIDF_ENGINE = None
_SEMANTIC_ENGINE = None


def _code_executor_tool(code: str, timeout: int = 30, user_id: str = "agent") -> Dict[str, Any]:
    """Run Python through the production CodeExecutor pipeline."""
    try:
        from code_executor import CodeExecutor

        executor = CodeExecutor(timeout=int(timeout), rate_limit=50, backend="subprocess")
        result = executor.execute(code, user_id=user_id)
        return result.to_dict()
    except Exception as e:
        return {"success": False, "error": f"CodeExecutor failed: {type(e).__name__}: {e}"}


def register_code_executor_tools(registry: ToolRegistry) -> None:
    """Register secure code execution tools, including a python override."""
    code_def = ToolDefinition(
        name="execute_python_secure",
        description=(
            "Execute Python code through the production security pipeline: rate limit, "
            "quick check, AST static analysis, sandbox execution, output sanitization, and audit logging."
        ),
        parameters=[
            ParameterSchema("code", "string", "Python source code to execute", required=True),
            ParameterSchema("timeout", "integer", "Execution timeout in seconds", required=False, default=30, min_value=1, max_value=60),
            ParameterSchema("user_id", "string", "User/actor id for rate limiting and audit", required=False, default="agent"),
        ],
        return_type="dict",
        return_description="CodeExecutionResult as a dictionary with success/output/error/analysis/audit fields",
        category="code",
        version="1.0",
    )
    registry.register(code_def, _code_executor_tool)

    # Override legacy python with the secure implementation so existing prompts can keep using "python".
    python_def = ToolDefinition(
        name="python",
        description="Execute Python code securely using CodeExecutor. Prefer for calculations, tests, and safe snippets.",
        parameters=code_def.parameters,
        return_type="dict",
        return_description=code_def.return_description,
        category="code",
        version="3.0",
    )
    registry.register(python_def, _code_executor_tool, warn_on_overwrite=False)


def _get_schema_fs():
    global _SCHEMA_FS
    if _SCHEMA_FS is None:
        from schema_aware_fs import SchemaAwareFS
        _SCHEMA_FS = SchemaAwareFS()
    return _SCHEMA_FS


def _schema_fs_write(path: str, content: str) -> Dict[str, Any]:
    return _get_schema_fs().write(path, content)


def _schema_fs_read(path: str) -> Dict[str, Any]:
    return _get_schema_fs().read(path)


def _schema_fs_list(prefix: str = "") -> Dict[str, Any]:
    return _get_schema_fs().list_files(prefix)


def _schema_fs_stats() -> Dict[str, Any]:
    result = _get_schema_fs().stats()
    # FileMetadata objects are not JSON-friendly in most_accessed; compact them.
    if result.get("success") and "most_accessed" in result:
        result["most_accessed"] = [m.to_dict() for m in result["most_accessed"]]
    return result


def _data_read(content: str = None, path: str = None) -> Dict[str, Any]:
    from unified_data_tools import DataReader
    if path:
        fs_result = _schema_fs_read(path)
        return DataReader.read(fs_result)
    if content is None:
        return {"success": False, "error": "Provide either 'content' or 'path'."}
    return DataReader.read(content)


def _data_query(data: dict, conditions: list = None, sort_by: Any = None,
                group_by: dict = None, select: list = None, limit: int = None) -> Dict[str, Any]:
    from unified_data_tools import DataQuery
    return DataQuery.query(data, conditions=conditions, sort_by=sort_by, group_by=group_by, select=select, limit=limit)


def _data_stats(data: dict, columns: list = None) -> Dict[str, Any]:
    from unified_data_tools import DataStats
    return DataStats.summarize(data, columns=columns)


def _data_derive(data: dict, new_column: str, expression: str) -> Dict[str, Any]:
    from unified_data_tools import DataTransform
    return DataTransform.derive(data, new_column, expression)


def register_data_tools(registry: ToolRegistry) -> None:
    """Register schema-aware FS and unified data tools."""
    registry.register(
        ToolDefinition(
            name="schema_fs_write",
            description="Write content to the in-memory schema-aware filesystem and infer CSV/JSON schema.",
            parameters=[ParameterSchema("path", "string", "Virtual path"), ParameterSchema("content", "string", "File content")],
            return_type="dict",
            return_description="Write status, content type, and whether schema was inferred",
            category="data",
        ),
        _schema_fs_write,
    )
    registry.register(
        ToolDefinition(
            name="schema_fs_read",
            description="Read a file from the in-memory schema-aware filesystem with metadata.",
            parameters=[ParameterSchema("path", "string", "Virtual path")],
            return_type="dict",
            return_description="Content plus metadata/schema summary",
            category="data",
        ),
        _schema_fs_read,
    )
    registry.register(
        ToolDefinition(
            name="schema_fs_list",
            description="List files in the in-memory schema-aware filesystem.",
            parameters=[ParameterSchema("prefix", "string", "Optional path prefix", required=False, default="")],
            return_type="dict",
            return_description="Matching file paths",
            category="data",
        ),
        _schema_fs_list,
    )
    registry.register(
        ToolDefinition(
            name="schema_fs_stats",
            description="Return global stats for the schema-aware filesystem.",
            parameters=[],
            return_type="dict",
            return_description="Filesystem stats and most accessed files",
            category="data",
        ),
        _schema_fs_stats,
    )
    registry.register(
        ToolDefinition(
            name="data_read",
            description="Parse CSV or JSON from raw content or from a schema_fs path.",
            parameters=[
                ParameterSchema("content", "string", "Raw CSV/JSON content", required=False),
                ParameterSchema("path", "string", "schema_fs path", required=False),
            ],
            return_type="dict",
            return_description="Typed tabular/json data with preview",
            category="data",
        ),
        _data_read,
    )
    registry.register(
        ToolDefinition(
            name="data_query",
            description="Filter/sort/group/select/limit a tabular data dict returned by data_read.",
            parameters=[
                ParameterSchema("data", "dict", "Data dict from data_read"),
                ParameterSchema("conditions", "list", "Filter conditions", required=False),
                ParameterSchema("sort_by", "list", "Sort specification", required=False),
                ParameterSchema("group_by", "dict", "Group-by specification", required=False),
                ParameterSchema("select", "list", "Columns to project", required=False),
                ParameterSchema("limit", "integer", "Max rows", required=False),
            ],
            return_type="dict",
            return_description="Query result rows and summary",
            category="data",
        ),
        _data_query,
    )
    registry.register(
        ToolDefinition(
            name="data_stats",
            description="Compute statistics over tabular data returned by data_read/data_query.",
            parameters=[
                ParameterSchema("data", "dict", "Data dict"),
                ParameterSchema("columns", "list", "Optional columns", required=False),
            ],
            return_type="dict",
            return_description="Column summaries/statistics",
            category="data",
        ),
        _data_stats,
    )
    registry.register(
        ToolDefinition(
            name="data_derive",
            description="Create a derived column in tabular data using a safe expression over existing columns.",
            parameters=[
                ParameterSchema("data", "dict", "Data dict"),
                ParameterSchema("new_column", "string", "Name of new column"),
                ParameterSchema("expression", "string", "Expression using column names"),
            ],
            return_type="dict",
            return_description="Data with derived column",
            category="data",
        ),
        _data_derive,
    )


def _local_tfidf_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    global _TFIDF_ENGINE
    try:
        if _TFIDF_ENGINE is None:
            from local_search import LocalTFIDFSearchEngine
            _TFIDF_ENGINE = LocalTFIDFSearchEngine()
            _TFIDF_ENGINE.index()
        return {"success": True, "results": _TFIDF_ENGINE.search(query, top_k=int(top_k))}
    except Exception as e:
        return {"success": False, "error": f"Local TF-IDF search failed: {type(e).__name__}: {e}"}


def _semantic_local_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    global _SEMANTIC_ENGINE
    try:
        if _SEMANTIC_ENGINE is None:
            from semantic_local_search import SemanticLocalSearchEngine
            _SEMANTIC_ENGINE = SemanticLocalSearchEngine()
        return {"success": True, "results": _SEMANTIC_ENGINE.search(query, top_k=int(top_k))}
    except Exception as e:
        return {"success": False, "error": f"Semantic search unavailable or failed: {type(e).__name__}: {e}"}


def _hybrid_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    try:
        from hybrid_search import HybridSearchAgent
        agent = HybridSearchAgent()
        results = agent.search(query, top_k=int(top_k))
        return {"success": True, "results": [r.__dict__ if hasattr(r, "__dict__") else r for r in results]}
    except Exception as e:
        return {"success": False, "error": f"Hybrid search unavailable or failed: {type(e).__name__}: {e}"}


def register_search_tools(registry: ToolRegistry) -> None:
    """Register live and local search tools."""
    # Live Tavily web search. Registration itself is safe without an API key;
    # runtime calls return a helpful error if no key is configured.
    try:
        from web_search_tools import register_tavily_tools
        register_tavily_tools(registry)
    except Exception as e:
        print(f"Warning: Tavily web_search registration skipped: {e}")

    common = [
        ParameterSchema("query", "string", "Search query", required=True),
        ParameterSchema("top_k", "integer", "Number of results", required=False, default=5, min_value=1, max_value=10),
    ]
    registry.register(
        ToolDefinition("local_tfidf_search", "Search the Notebook 15 local corpus with TF-IDF keyword ranking.", common, "dict", "Ranked local results", category="search"),
        _local_tfidf_search,
    )
    registry.register(
        ToolDefinition("semantic_local_search", "Search the Notebook 15 local corpus with BGE/FAISS semantic retrieval when available.", common, "dict", "Ranked semantic local results", category="search"),
        _semantic_local_search,
    )
    registry.register(
        ToolDefinition("hybrid_search", "Fuse web, local TF-IDF, and local semantic search results when dependencies/API keys are available.", common, "dict", "Merged hybrid search results", category="search"),
        _hybrid_search,
    )


def build_full_tool_registry(include_search: bool = True, include_data: bool = True,
                             include_code_executor: bool = True) -> ToolRegistry:
    """Build the complete registry used by AgentRuntime.

    Starts from build_production_registry() (base + production + stateful tools), then
    overlays secure code execution, data tools, and search integrations.
    """
    registry = build_production_registry()
    if include_code_executor:
        register_code_executor_tools(registry)
    if include_data:
        register_data_tools(registry)
    if include_search:
        register_search_tools(registry)
    return registry


if __name__ == "__main__":
    reg = build_full_tool_registry()
    print(f"Registered {len(reg.tools)} tools:")
    for name in sorted(reg.tools):
        print(" -", name)
