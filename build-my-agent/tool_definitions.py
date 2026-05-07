"""
Tool Definitions — Formal schemas for production tools.

Replaces the informal 'params' dicts in tools.py with structured dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json


@dataclass
class ParameterSchema:
    """Schema for a single tool parameter."""
    name: str
    type: str           # "string", "number", "integer", "boolean", "list", "dict"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None     # allowed values
    min_value: Optional[float] = None    # for numbers
    max_value: Optional[float] = None    # for numbers
    pattern: Optional[str] = None        # regex for strings

    def to_dict(self) -> dict:
        """Convert to JSON-schema compatible dict."""
        d = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required
        }
        if self.default is not None:
            d["default"] = self.default
        if self.enum:
            d["enum"] = self.enum
        if self.min_value is not None:
            d["min_value"] = self.min_value
        if self.max_value is not None:
            d["max_value"] = self.max_value
        if self.pattern:
            d["pattern"] = self.pattern
        return d


@dataclass
class ToolDefinition:
    """Complete definition for a production tool."""
    name: str
    description: str
    parameters: List[ParameterSchema]
    return_type: str       # "string", "number", "dict", "list", etc.
    return_description: str
    examples: List[Dict[str, Any]] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0"

    def to_schema_dict(self) -> dict:
        """Convert to JSON schema for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": {
                "type": self.return_type,
                "description": self.return_description
            },
            "examples": self.examples,
            "category": self.category,
            "version": self.version,
        }

    def to_prompt_string(self) -> str:
        """Format for inclusion in system prompts."""
        lines = [
            f"### {self.name}",
            f"{self.description}\n",
            "Parameters:"
        ]
        
        for p in self.parameters:
            req = "required" if p.required else f"optional, default={p.default}"
            lines.append(f"  - {p.name} ({p.type}, {req}): {p.description}")
            if p.enum:
                lines.append(f"    Allowed values: {p.enum}")
            if p.min_value is not None or p.max_value is not None:
                lines.append(f"    Range: [{p.min_value}, {p.max_value}]")
        
        lines.append(f"\nReturns ({self.return_type}): {self.return_description}")
        
        if self.examples:
            lines.append("\nExamples:")
            for ex in self.examples:
                lines.append(f'  Input:  {json.dumps(ex.get("input", {}))}')
                lines.append(f'  Output: {json.dumps(ex.get("output", ""))}')
        
        return "\n".join(lines)


# ============================================================================
# Example: Convert your existing 'python' tool to a formal definition
# ============================================================================

PYTHON_TOOL_DEF = ToolDefinition(
    name="python",
    description="Execute Python code and return the output. Use for calculations, data manipulation, and testing ideas.",
    parameters=[
        ParameterSchema(
            name="code",
            type="string",
            description="The Python code to execute",
            required=True,
        ),
        ParameterSchema(
            name="timeout",
            type="number",
            description="Max seconds to run (default 10, max 30)",
            required=False,
            default=10,
            min_value=1,
            max_value=30,
        ),
    ],
    return_type="string",
    return_description="The stdout of the executed code, or an error message",
    examples=[
        {
            "input": {"code": "2 ** 10"},
            "output": "1024"
        },
        {
            "input": {"code": "import math\nmath.sqrt(144)"},
            "output": "12.0"
        },
    ],
    category="code",
    version="2.0",  # Bumped from your original
)


if __name__ == "__main__":
    print("=== ToolDefinition Example ===")
    print(json.dumps(PYTHON_TOOL_DEF.to_schema_dict(), indent=2))
    print("\n=== Prompt Format ===")
    print(PYTHON_TOOL_DEF.to_prompt_string())