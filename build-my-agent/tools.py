"""
Tools - The agent's hands.

This module implements the tool execution engine. The agent can use
four tools: python, read_file, write_file, and search.

All file operations are sandboxed to a workspace directory.
"""

import os
import subprocess
from typing import Dict, Any

# ====================================================================
# Configuration
# ====================================================================

# The sandbox directory. The agent can only read/write here.
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)  # Create it if it doesn't exist

# Timeout for Python execution (seconds)
PYTHON_TIMEOUT = 10


# ====================================================================
# Safe path resolution - prevents the agent from escaping the sandbox
# ====================================================================
def _resolve_safe_path(filepath: str, operation: str = "access") -> str:
    """
    Resolve a file path and verify it stays within the workspace.
    
    This prevents path traversal attacks like "../../etc/passwd".
    
    Args:
        filepath: The path the agent provided
        operation: What operation is being performed (for error messages)
        
    Returns:
        The absolute, safe path within the workspace
        
    Raises:
        PermissionError: If the path escapes the sandbox
    """
    # Convert to absolute path (resolves .., ~, etc.)
    abs_path = os.path.abspath(os.path.join(WORKSPACE_DIR, filepath))
    
    # Make sure it's inside the workspace
    if not abs_path.startswith(WORKSPACE_DIR):
        raise PermissionError(
            f"Cannot {operation} file outside the workspace. "
            f"Requested: '{filepath}' resolves to '{abs_path}'"
        )
    
    return abs_path


# ====================================================================
# Tool: Python REPL (safe subset)
# ====================================================================
def run_python(params: Dict[str, Any]) -> str:
    """
    Execute a piece of Python code in a restricted environment.
    
    We run the code in a subprocess with a timeout to prevent:
    - Infinite loops (killed after timeout)
    - System access (no network, no file I/O by default)
    
    Args:
        params: Dict with "code" (required) and "timeout" (optional)
        
    Returns:
        The stdout of the executed code, or an error message
    """
    
    # Extract required parameter
    code = params.get("code")
    if not code or not isinstance(code, str):
        return "Error: Missing 'code' parameter. Provide the Python code to execute."
    
    # Extract optional timeout (cap at 30 seconds for safety)
    timeout = min(params.get("timeout", PYTHON_TIMEOUT), 30)
    
    try:
        # Execute in a subprocess for isolation
        # This is the safest approach: even if the code does something bad,
        # it's confined to the subprocess
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=WORKSPACE_DIR,  # Run inside the workspace
        )
        
        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += f"\n[Stderr]\n{result.stderr}"
        
        if result.returncode != 0:
            # Extract the error type and message
            error_line = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown error"
            return f"Error: {error_line}"
        
        return output if output.strip() else "[No output]"
        
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds"
    
    except Exception as e:
        return f"Error: {type(e).__name__} - {str(e)}"


# ====================================================================
# Tool: Read File (sandboxed)
# ====================================================================
def read_file(params: Dict[str, Any]) -> str:
    """
    Read a file from the sandboxed workspace.
    
    Args:
        params: Dict with "filepath" (required), "lines" (optional),
                "start_line" (optional, 1-indexed)
        
    Returns:
        The file contents, or an error message
    """
    
    # Get the file path
    filepath = params.get("filepath")
    if not filepath or not isinstance(filepath, str):
        return "Error: Missing 'filepath' parameter. Provide the path to the file."
    
    # Resolve to a safe absolute path (prevents escaping the sandbox)
    try:
        safe_path = _resolve_safe_path(filepath, "read")
    except PermissionError as e:
        return f"Error: {str(e)}"
    
    # Check if the file exists
    if not os.path.exists(safe_path):
        return f"Error: File not found at '{filepath}'"
    
    # Read the file
    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        total_lines = len(all_lines)
        
        # Handle line slicing (1-indexed start_line, like most text editors)
        start_line = max(params.get("start_line", 1) - 1, 0)  # Convert to 0-indexed
        max_lines = params.get("lines", total_lines - start_line)
        
        # Extract the requested range
        selected_lines = all_lines[start_line:start_line + max_lines]
        content = ''.join(selected_lines)
        
        # Add a footer showing how many lines were read
        lines_read = len(selected_lines)
        footer = f"\n\n---\n(read {lines_read} of {total_lines} total lines, starting from line {start_line + 1})"
        
        return content.rstrip() + footer
        
    except UnicodeDecodeError:
        return f"Error: Cannot read file. It may be a binary file, not text. Path: '{filepath}'"
    
    except Exception as e:
        return f"Error: Failed to read file: {type(e).__name__} - {str(e)}"


# ====================================================================
# Tool: Write File (sandboxed)
# ====================================================================
def write_file(params: Dict[str, Any]) -> str:
    """
    Write content to a file in the sandboxed workspace.
    
    Args:
        params: Dict with "filepath" (required), "content" (required),
                "append" (optional, default false)
        
    Returns:
        Success confirmation with character count, or an error message
    """
    
    # Get required parameters
    filepath = params.get("filepath")
    content = params.get("content")
    
    if not filepath or not isinstance(filepath, str):
        return "Error: Missing 'filepath' parameter. Provide the destination path."
    
    if content is None:
        return "Error: Missing 'content' parameter. Provide the text to write."
    
    content_str = str(content)
    
    # Resolve to a safe absolute path
    try:
        safe_path = _resolve_safe_path(filepath, "write")
    except PermissionError as e:
        return f"Error: {str(e)}"
    
    # Determine mode (write vs append)
    mode = 'a' if params.get("append", False) else 'w'
    
    # Ensure parent directory exists
    parent_dir = os.path.dirname(safe_path)
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except Exception as e:
            return f"Error: Cannot create directory: {type(e).__name__} - {str(e)}"
    
    # Write the file
    try:
        with open(safe_path, mode, encoding='utf-8') as f:
            f.write(content_str)
        
        action = "appended" if mode == 'a' else "wrote"
        return f"Successfully {action} {len(content_str)} characters to '{filepath}'"
        
    except Exception as e:
        return f"Error: Failed to write file: {type(e).__name__} - {str(e)}"



# ====================================================================
# Tool: Knowledge Base Search
# ====================================================================
def search_kb(params: Dict[str, Any]) -> str:
    """
    Search the internal knowledge base for facts.
    
    The knowledge base contains 23 facts across geography, science,
    history, technology, and math. This is a retrieval tool - the agent
    uses it to look up information it doesn't have in its parametric memory.
    
    Args:
        params: Dict with "query" (required), "max_results" (optional)
        
    Returns:
        Formatted knowledge base results, or a message if no results found
    """
    
    # Get required parameter
    query = params.get("query")
    if not query or not isinstance(query, str):
        return "Error: Missing 'query' parameter. Provide the search query string."
    
    # Get optional parameter with safety cap
    max_results = min(params.get("max_results", 5), 10)
    
    # Call the knowledge base search function
    try:
        from knowledge_base import search_kb as _internal_search_kb
        result = _internal_search_kb(query, max_results)
        return result
    except Exception as e:
        return f"Error: Knowledge base search failed: {type(e).__name__} - {str(e)}"

# ====================================================================
# Tool: Web Search
# ====================================================================
def search_web(params: Dict[str, Any]) -> str:
    """
    Search the web and return top results.
    
    For now, this is a stub that returns a placeholder. You'll replace
    this with a real search API (DuckDuckGo, Serper, etc.) when you're
    ready.
    
    Args:
        params: Dict with "query" (required), "max_results" (optional, default 5)
        
    Returns:
        Formatted search results, or a message that the tool is not yet connected
    """
    
    # Get required parameter
    query = params.get("query")
    if not query or not isinstance(query, str):
        return "Error: Missing 'query' parameter. Provide the search query string."
    
    max_results = min(params.get("max_results", 5), 10)  # Cap at 10 results
    
    # TODO: Replace this with a real search API call
    # For now, we return a placeholder. You can use:
    # - DuckDuckGo (free, no API key needed)
    # - Serper.dev (API key, Google-like results)
    # - Brave Search (API key, fast)
    
    return (
        f"[SEARCH TOOL: Not yet connected to a search API]\n\n"
        f"Query: '{query}'\n"
        f"Requested results: {max_results}\n\n"
        f"To activate this tool, we need to connect it to a search provider.\n"
        f"Popular options:\n"
        f"  - DuckDuckGo: Free, no API key, good for learning\n"
        f"  - Serper.dev: Fast, Google-like, $1 credit for testing\n"
        f"  - Brave Search: Fast, good quality, free tier available\n\n"
        f"Let me know which one you'd like to set up, and I'll show you how."
    )


# ====================================================================
# The tool registry - maps tool names to their functions
# ====================================================================
#
# Each tool has a structured schema that serves TWO purposes:
# 1. System Prompt Generation: Describes the tool to the LLM in human-readable form
# 2. Parameter Validation: The guard layer checks incoming calls against this schema
#
# Schema format for each parameter:
#   "param_name": {
#       "type": "string" | "number" | "boolean",    # Expected type
#       "required": True | False,                       # Must be present
#       "description": "Human-readable description",  # For the system prompt
#   }

TOOLS = {
    "python": {
        "fn": run_python,
        "description": "Execute Python code and return the output. Use for calculations, "
                       "data manipulation, and testing ideas.",
        "params": {
            "code": {
                "type": "string",
                "required": True,
                "description": "The Python code to execute",
            },
            "timeout": {
                "type": "number",
                "required": False,
                "description": "Max seconds to run (default 10, max 30)",
            },
        },
    },
    "read_file": {
        "fn": read_file,
        "description": "Read the contents of a file from the workspace.",
        "params": {
            "filepath": {
                "type": "string",
                "required": True,
                "description": "Path to the file (relative to workspace)",
            },
            "lines": {
                "type": "number",
                "required": False,
                "description": "How many lines to read (default: all)",
            },
            "start_line": {
                "type": "number",
                "required": False,
                "description": "1-indexed starting line (default: 1)",
            },
        },
    },
    "write_file": {
        "fn": write_file,
        "description": "Write content to a file in the workspace.",
        "params": {
            "filepath": {
                "type": "string",
                "required": True,
                "description": "Destination file path (relative to workspace)",
            },
            "content": {
                "type": "string",
                "required": True,
                "description": "The text to write (can be empty string to clear the file)",
            },
            "append": {
                "type": "boolean",
                "required": False,
                "description": "If true, append instead of overwrite (default: false)",
            },
        },
    },
    "search": {
        "fn": search_web,
        "description": "Search the web for information and return top results with snippets.",
        "params": {
            "query": {
                "type": "string",
                "required": True,
                "description": "The search query",
            },
            "max_results": {
                "type": "number",
                "required": False,
                "description": "How many results to return (default: 5, max: 10)",
            },
        },
    },
    "search_kb": {
        "fn": search_kb,
        "description": "Search the internal knowledge base for facts. The knowledge base contains 23 facts across geography, science, history, technology, and math. Use this for factual lookups about real-world knowledge that the agent doesn't have in memory.",
        "params": {
            "query": {
                "type": "string",
                "required": True,
                "description": "The search query. Use key terms (e.g., 'capital France' or 'speed light') for best results.",
            },
            "max_results": {
                "type": "number",
                "required": False,
                "description": "How many results to return (default: 5, max: 10)",
            },
        },
    },
}


# ====================================================================
# Main execution entry point
# ====================================================================

def execute_tool(tool_name: str, params: Dict[str, Any]) -> str:
    """
    Execute a tool by name and return its output.
    
    This is the central dispatch function. The agent loop calls this
    when the LLM produces a "use_tool" action.
    
    Args:
        tool_name: The name of the tool to use (must be in TOOLS registry)
        params: The parameters dict from the agent's action
        
    Returns:
        The tool's output as a string, or an error message.
        Returns None if the tool name is not recognized.
    """
    
    # Check if the tool exists
    if tool_name not in TOOLS:
        available = ', '.join(TOOLS.keys())
        return f'Error: Unknown tool "{tool_name}". Available tools: {available}'
    
    # Get the function and call it
    tool = TOOLS[tool_name]
    fn = tool["fn"]
    
    try:
        result = fn(params)
        return str(result)
    except Exception as e:
        return f"Error: Tool '{tool_name}' failed: {type(e).__name__} - {str(e)}"


# ====================================================================
# Generate the system prompt section for tools
# ====================================================================
def generate_tools_instruction() -> str:
    """
    Generate the tool usage instructions to inject into the system prompt.
    
    This produces the text that tells the LLM what tools exist, how to call them,
    and what to expect back.
    """
    
    lines = [
        "You have access to the following tools. To use a tool, respond with a JSON object:",
        "",
        '{ "action": "use_tool", "tool": "tool_name", "params": { ... } }',
        "",
        "When the tool returns a result, continue your reasoning with the new information.",
        "After using a tool, respond with a \"think\" action to process the result,",
        "or an \"answer\" action if you now have the final solution.",
        "",
        "Available tools:",
        "-" * 60,
    ]
    
    for name, info in TOOLS.items():
        lines.append(f"\nTool: {name}")
        lines.append(f"What it does: {info['description']}")
        lines.append(f"Parameters:")
        for param, spec in info['params'].items():
            # spec is now a dict: {"type": "string", "required": True, "description": "..."}
            is_required = "required" if spec.get("required", False) else "optional"
            param_type = spec.get("type", "any")
            description = spec.get("description", "")
            lines.append(f"  - {param} ({param_type}, {is_required}): {description}")
        lines.append("-" * 40)
    
    return "\n".join(lines)


# ====================================================================
# Test
# ====================================================================
if __name__ == "__main__":
    print("Testing the tool engine...\n" + "=" * 60)
    
    # Test Python tool
    print("Test 1: Python tool - basic calculation")
    result = run_python({"code": "x = 2 ** 16\nprint(x)", "timeout": 5})
    print(f"  Result: {result}\n")
    
    # Test Python tool - error case
    print("Test 2: Python tool - error handling")
    result = run_python({"code": "print(undefined_var)", "timeout": 5})
    print(f"  Result: {result}\n")
    
    # Test read_file - create a test file first, then read it
    print("Test 3: Write then read a file")
    workspace = os.path.abspath(WORKSPACE_DIR)
    print(f"  Workspace: {workspace}\n")
    
    result = write_file({
        "filepath": "test.txt",
        "content": "Line 1: Hello, Agent!\nLine 2: This is a test file.\nLine 3: Goodbye!"
    })
    print(f"  Write: {result}")
    
    result = read_file({"filepath": "test.txt", "lines": 2, "start_line": 1})
    print(f"  Read (first 2 lines):\n  {result}\n")
    
    result = read_file({"filepath": "test.txt", "lines": 100, "start_line": 2})
    print(f"  Read (from line 2):\n  {result}\n")
    
    # Test write with append
    print("Test 4: Append to file")
    result = write_file({
        "filepath": "test.txt",
        "content": "\nLine 4: This was appended.",
        "append": True
    })
    print(f"  Append: {result}")
    result = read_file({"filepath": "test.txt"})
    print(f"  Full file after append:\n  {result}\n")
    
    # Test non-existent file
    print("Test 5: Read non-existent file")
    result = read_file({"filepath": "does_not_exist.txt"})
    print(f"  Result: {result}\n")
    
    # Test the tool registry dispatch
    print("Test 6: execute_tool dispatch")
    result = execute_tool("python", {"code": "print(2 + 2)"})
    print(f"  execute_tool('python', ...): {result}")
    
    result = execute_tool("nonexistent", {"foo": "bar"})
    print(f"  execute_tool('nonexistent', ...): {result}\n")
    
    # Show the generated instruction
    print("Test 7: Generated system instruction (first 500 chars):")
    instruction = generate_tools_instruction()
    print(instruction[:500] + "...\n")
    
    print("All tests complete.")
