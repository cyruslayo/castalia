"""
Production Tools — 10 additional tools for the advanced tool registry.

These tools extend the registry beyond the basic tools from tools.py,
demonstrating advanced design patterns: safe calculators, string utils,
data validators, format converters, encoding tools, and more.

Each tool follows the 5 design principles from Notebook 13:
1. Single Responsibility
2. Clear Schemas
3. Predictable Errors
4. Usage Examples
5. Fail Gracefully
"""

import json
import re
import math
import csv
import io
import base64
import hashlib
import urllib.parse
import statistics
from functools import reduce
from typing import Dict, Any, List, Optional

from tool_definitions import ParameterSchema, ToolDefinition


# ============================================================================
# Tool 1: Safe Calculator
# ============================================================================

SAFE_OPS = {
    'sqrt': math.sqrt, 'abs': abs, 'round': round,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'log': math.log, 'log10': math.log10,
    'pi': math.pi, 'e': math.e,
    'pow': pow, 'min': min, 'max': max,
    'floor': math.floor, 'ceil': math.ceil,
}

def tool_calculator(expression: str) -> dict:
    """Safely evaluate a mathematical expression."""
    cleaned = expression.strip()
    # Remove known safe function names to check remaining chars
    check = cleaned
    for fname in SAFE_OPS:
        check = check.replace(fname, "")
    allowed_chars = set("0123456789+-*/().% ,")
    if not all(c in allowed_chars for c in check):
        return {"success": False, "error": f"Expression contains disallowed characters. Only math operations are permitted."}
    try:
        result = eval(expression, {"__builtins__": {}}, SAFE_OPS)
        return {"success": True, "result": result, "expression": expression}
    except Exception as e:
        return {"success": False, "error": f"Math error: {e}. Check your expression syntax."}

CALCULATOR_DEF = ToolDefinition(
    name="calculator",
    description="Safely evaluate math expressions. Supports +,-,*,/,**,sqrt,sin,cos,tan,log,log10,pi,e,abs,round,pow,min,max,floor,ceil.",
    parameters=[
        ParameterSchema("expression", "string", "Math expression like '2+3*4' or 'sqrt(144)'"),
    ],
    return_type="dict",
    return_description="{'success': bool, 'result': number} or {'success': False, 'error': str}",
    examples=[
        {"input": {"expression": "2 + 3 * 4"}, "output": {"success": True, "result": 14}},
        {"input": {"expression": "sqrt(144) + pi"}, "output": {"success": True, "result": 15.14159}},
    ],
    category="math",
    version="1.0",
)


# ============================================================================
# Tool 2: String Utilities
# ============================================================================

def tool_string_utils(text: str, operation: str) -> dict:
    """Perform string operations."""
    ops = {
        "upper": lambda t: t.upper(),
        "lower": lambda t: t.lower(),
        "title": lambda t: t.title(),
        "reverse": lambda t: t[::-1],
        "length": lambda t: len(t),
        "word_count": lambda t: len(t.split()),
        "strip": lambda t: t.strip(),
        "char_count": lambda t: dict(sorted(((c, t.count(c)) for c in set(t) if c.strip()), key=lambda x: -x[1])[:10]),
    }
    if operation not in ops:
        return {"success": False, "error": f"Unknown operation '{operation}'. Available: {list(ops.keys())}"}
    try:
        result = ops[operation](text)
        return {"success": True, "operation": operation, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

STRING_UTILS_DEF = ToolDefinition(
    name="string_utils",
    description="Perform text operations: upper, lower, title, reverse, length, word_count, strip, char_count.",
    parameters=[
        ParameterSchema("text", "string", "Input text"),
        ParameterSchema("operation", "string", "Operation to perform",
                        enum=["upper", "lower", "title", "reverse", "length", "word_count", "strip", "char_count"]),
    ],
    return_type="dict",
    return_description="{'success': bool, 'operation': str, 'result': ...}",
    examples=[
        {"input": {"text": "Hello World", "operation": "word_count"}, "output": {"success": True, "result": 2}},
    ],
    category="text",
    version="1.0",
)


# ============================================================================
# Tool 3: List Operations
# ============================================================================

def tool_list_ops(items: list, operation: str) -> dict:
    """Perform operations on lists."""
    ops = {
        "sort": lambda lst: sorted(lst),
        "reverse": lambda lst: list(reversed(lst)),
        "unique": lambda lst: list(dict.fromkeys(lst)),
        "length": lambda lst: len(lst),
        "sum": lambda lst: sum(lst) if all(isinstance(x, (int, float)) for x in lst) else "Error: not all items are numbers",
        "min": lambda lst: min(lst) if lst else "Error: empty list",
        "max": lambda lst: max(lst) if lst else "Error: empty list",
        "flatten": lambda lst: [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])],
        "frequencies": lambda lst: dict(sorted(((x, lst.count(x)) for x in set(map(str, lst))), key=lambda p: -p[1])),
    }
    if operation not in ops:
        return {"success": False, "error": f"Unknown operation '{operation}'. Available: {list(ops.keys())}"}
    try:
        result = ops[operation](items)
        if isinstance(result, str) and result.startswith("Error"):
            return {"success": False, "error": result}
        return {"success": True, "operation": operation, "result": result, "input_length": len(items)}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}

LIST_OPS_DEF = ToolDefinition(
    name="list_ops",
    description="List operations: sort, reverse, unique, length, sum, min, max, flatten, frequencies.",
    parameters=[
        ParameterSchema("items", "list", "The list to operate on"),
        ParameterSchema("operation", "string", "Operation name",
                        enum=["sort", "reverse", "unique", "length", "sum", "min", "max", "flatten", "frequencies"]),
    ],
    return_type="dict",
    return_description="{'success': bool, 'result': ...}",
    category="data",
    version="1.0",
)


# ============================================================================
# Tool 4: Dictionary Operations
# ============================================================================

def tool_dict_ops(data: dict, operation: str, key: str = None) -> dict:
    """Perform operations on dictionaries."""
    ops_no_key = {
        "keys": lambda d, _: list(d.keys()),
        "values": lambda d, _: list(d.values()),
        "length": lambda d, _: len(d),
        "sorted_keys": lambda d, _: dict(sorted(d.items())),
        "invert": lambda d, _: {str(v): k for k, v in d.items()},
    }
    ops_with_key = {
        "get": lambda d, k: d.get(k, f"Key '{k}' not found. Available: {list(d.keys())}"),
        "has_key": lambda d, k: k in d,
        "delete": lambda d, k: {dk: dv for dk, dv in d.items() if dk != k},
    }
    all_ops = {**ops_no_key, **ops_with_key}
    if operation not in all_ops:
        return {"success": False, "error": f"Unknown operation '{operation}'. Available: {list(all_ops.keys())}"}
    if operation in ops_with_key and key is None:
        return {"success": False, "error": f"Operation '{operation}' requires a 'key' parameter."}
    try:
        result = all_ops[operation](data, key)
        return {"success": True, "operation": operation, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

DICT_OPS_DEF = ToolDefinition(
    name="dict_ops",
    description="Dictionary operations: keys, values, length, sorted_keys, invert, get, has_key, delete.",
    parameters=[
        ParameterSchema("data", "dict", "The dictionary to operate on"),
        ParameterSchema("operation", "string", "Operation name",
                        enum=["keys", "values", "length", "sorted_keys", "invert", "get", "has_key", "delete"]),
        ParameterSchema("key", "string", "Key for get/has_key/delete operations", required=False),
    ],
    return_type="dict",
    return_description="{'success': bool, 'result': ...}",
    category="data",
    version="1.0",
)


# ============================================================================
# Tool 5: Date & Time
# ============================================================================

from datetime import datetime, timedelta

def tool_date_time(operation: str, date_str: str = None, days: int = None, fmt: str = None) -> dict:
    """Date and time operations."""
    try:
        if operation == "now":
            now = datetime.now()
            return {"success": True, "result": now.strftime("%Y-%m-%d %H:%M:%S"), "timestamp": now.timestamp()}
        elif operation == "parse":
            if not date_str:
                return {"success": False, "error": "Operation 'parse' requires 'date_str'. Example: date_str='2024-01-15'"}
            f = fmt or "%Y-%m-%d"
            dt = datetime.strptime(date_str, f)
            return {"success": True, "parsed": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "weekday": dt.strftime("%A"), "day_of_year": dt.timetuple().tm_yday}
        elif operation == "add_days":
            if not date_str:
                return {"success": False, "error": "Operation 'add_days' requires 'date_str'."}
            if days is None:
                return {"success": False, "error": "Operation 'add_days' requires 'days' parameter."}
            dt = datetime.strptime(date_str, fmt or "%Y-%m-%d")
            result_dt = dt + timedelta(days=days)
            return {"success": True, "original": date_str, "days_added": days,
                    "result": result_dt.strftime("%Y-%m-%d"), "weekday": result_dt.strftime("%A")}
        elif operation == "diff":
            if not date_str or not fmt:
                return {"success": False, "error": "Operation 'diff' requires 'date_str' (first date) and 'fmt' (second date in YYYY-MM-DD)."}
            dt1 = datetime.strptime(date_str, "%Y-%m-%d")
            dt2 = datetime.strptime(fmt, "%Y-%m-%d")
            diff = abs((dt2 - dt1).days)
            return {"success": True, "date1": date_str, "date2": fmt, "difference_days": diff}
        else:
            return {"success": False, "error": f"Unknown operation '{operation}'. Available: now, parse, add_days, diff"}
    except ValueError as e:
        return {"success": False, "error": f"Date parsing error: {e}. Use format YYYY-MM-DD."}

DATE_TIME_DEF = ToolDefinition(
    name="date_time",
    description="Date/time operations: now, parse, add_days, diff.",
    parameters=[
        ParameterSchema("operation", "string", "Operation: now, parse, add_days, diff",
                        enum=["now", "parse", "add_days", "diff"]),
        ParameterSchema("date_str", "string", "Date string (YYYY-MM-DD)", required=False),
        ParameterSchema("days", "integer", "Number of days to add", required=False),
        ParameterSchema("fmt", "string", "Date format string or second date for diff", required=False),
    ],
    return_type="dict",
    return_description="Operation result dict",
    category="utility",
    version="1.0",
)


# ============================================================================
# Tool 6: Text Statistics
# ============================================================================

def tool_text_stats(text: str) -> dict:
    """Compute comprehensive text statistics."""
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    word_lengths = [len(w) for w in words]
    return {
        "success": True,
        "characters": len(text),
        "characters_no_spaces": len(text.replace(" ", "")),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": text.count("\n\n") + 1,
        "avg_word_length": round(statistics.mean(word_lengths), 1) if word_lengths else 0,
        "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
        "longest_word": max(words, key=len) if words else "",
        "unique_words": len(set(w.lower().strip(".,!?;:") for w in words)),
        "lexical_diversity": round(len(set(w.lower().strip(".,!?;:") for w in words)) / max(len(words), 1), 3),
    }

TEXT_STATS_DEF = ToolDefinition(
    name="text_stats",
    description="Compute comprehensive text statistics: word count, sentence count, avg lengths, lexical diversity.",
    parameters=[
        ParameterSchema("text", "string", "Text to analyze"),
   ],
    return_type="dict",
    return_description="Statistics dictionary with counts, averages, and diversity metrics",
    category="text",
    version="1.0",
)


# ============================================================================
# Tool 7: Format Converter
# ============================================================================

def tool_format_converter(data: str, from_format: str, to_format: str) -> dict:
    """Convert data between formats."""
    try:
        # Parse input
        if from_format == "json":
            parsed = json.loads(data)
        elif from_format == "csv":
            reader = csv.DictReader(io.StringIO(data))
            parsed = list(reader)
        elif from_format == "text_lines":
            parsed = [line.strip() for line in data.strip().split("\n") if line.strip()]
        else:
            return {"success": False, "error": f"Unknown from_format '{from_format}'. Use: json, csv, text_lines"}

        # Convert to output format
        if to_format == "json":
            result = json.dumps(parsed, indent=2)
        elif to_format == "csv":
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=parsed[0].keys())
                writer.writeheader()
                writer.writerows(parsed)
                result = output.getvalue()
            else:
                return {"success": False, "error": "CSV output requires list of dicts"}
        elif to_format == "markdown_table":
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                headers = list(parsed[0].keys())
                lines = ["| " + " | ".join(headers) + " |",
                          "| " + " | ".join(["---"] * len(headers)) + " |"]
                for row in parsed:
                    lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
                result = "\n".join(lines)
            else:
                return {"success": False, "error": "Markdown table requires list of dicts"}
        else:
            return {"success": False, "error": f"Unknown to_format '{to_format}'. Use: json, csv, markdown_table"}

        return {"success": True, "result": result, "from": from_format, "to": to_format}
    except Exception as e:
        return {"success": False, "error": f"Conversion error: {e}"}

FORMAT_CONVERTER_DEF = ToolDefinition(
    name="format_converter",
    description="Convert data between formats: json, csv, text_lines, markdown_table.",
    parameters=[
        ParameterSchema("data", "string", "Input data as string"),
        ParameterSchema("from_format", "string", "Source format", enum=["json", "csv", "text_lines"]),
        ParameterSchema("to_format", "string", "Target format", enum=["json", "csv", "markdown_table"]),
    ],
    return_type="dict",
    return_description="{'success': bool, 'result': converted_string}",
    category="utility",
    version="1.0",
)


# ============================================================================
# Tool 8: Data Validator
# ============================================================================

def tool_data_validator(data: dict, rules: dict) -> dict:
    """Validate a data dictionary against rules."""
    errors = []
    warnings = []
    for field_name, field_rules in rules.items():
        value = data.get(field_name)
        # Required check
        if field_rules.get("required", False) and (value is None or value == ""):
            errors.append(f"Field '{field_name}' is required but missing or empty")
            continue
        if value is None:
            continue
        # Type check
        expected_type = field_rules.get("type")
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Field '{field_name}': expected string, got {type(value).__name__}")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"Field '{field_name}': expected number, got {type(value).__name__}")
        # Min/max
        if "min" in field_rules and isinstance(value, (int, float)) and value < field_rules["min"]:
            errors.append(f"Field '{field_name}': {value} < minimum {field_rules['min']}")
        if "max" in field_rules and isinstance(value, (int, float)) and value > field_rules["max"]:
            errors.append(f"Field '{field_name}': {value} > maximum {field_rules['max']}")
        # Pattern
        if "pattern" in field_rules and isinstance(value, str):
            if not re.match(field_rules["pattern"], value):
                errors.append(f"Field '{field_name}': '{value}' doesn't match pattern '{field_rules['pattern']}'")
        # Allowed values
        if "allowed" in field_rules and value not in field_rules["allowed"]:
            errors.append(f"Field '{field_name}': '{value}' not in {field_rules['allowed']}")
        # Min length
        if "min_length" in field_rules and isinstance(value, str) and len(value) < field_rules["min_length"]:
            warnings.append(f"Field '{field_name}': length {len(value)} < recommended minimum {field_rules['min_length']}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "fields_checked": len(rules),
    }

DATA_VALIDATOR_DEF = ToolDefinition(
    name="data_validator",
    description="Validate a data dict against rules (required, type, min/max, pattern, allowed values).",
    parameters=[
        ParameterSchema("data", "dict", "Data dictionary to validate"),
        ParameterSchema("rules", "dict", "Validation rules per field"),
    ],
    return_type="dict",
    return_description="{'valid': bool, 'errors': [...], 'warnings': [...]}",
    category="utility",
    version="1.0",
)


# ============================================================================
# Tool 9: Advanced Math
# ============================================================================

def tool_math_advanced(operation: str, numbers: list) -> dict:
    """Advanced math operations on number lists."""
    if not numbers:
        return {"success": False, "error": "Empty number list. Provide at least one number."}
    if not all(isinstance(n, (int, float)) for n in numbers):
        return {"success": False, "error": f"All items must be numbers. Got types: {[type(n).__name__ for n in numbers]}"}
    try:
        if operation == "mean":
            result = statistics.mean(numbers)
        elif operation == "median":
            result = statistics.median(numbers)
        elif operation == "stdev":
            if len(numbers) < 2:
                return {"success": False, "error": "Standard deviation requires at least 2 numbers"}
            result = statistics.stdev(numbers)
        elif operation == "variance":
            if len(numbers) < 2:
                return {"success": False, "error": "Variance requires at least 2 numbers"}
            result = statistics.variance(numbers)
        elif operation == "product":
            result = reduce(lambda a, b: a * b, numbers)
        elif operation == "gcd":
            int_nums = [int(n) for n in numbers]
            result = reduce(math.gcd, int_nums)
        elif operation == "lcm":
            def lcm(a, b): return abs(a * b) // math.gcd(a, b)
            int_nums = [int(n) for n in numbers]
            result = reduce(lcm, int_nums)
        elif operation == "percentile_25":
            sorted_nums = sorted(numbers)
            idx = len(sorted_nums) * 0.25
            result = sorted_nums[int(idx)]
        elif operation == "percentile_75":
            sorted_nums = sorted(numbers)
            idx = len(sorted_nums) * 0.75
            result = sorted_nums[min(int(idx), len(sorted_nums) - 1)]
        elif operation == "range":
            result = max(numbers) - min(numbers)
        else:
            return {"success": False, "error": f"Unknown operation '{operation}'. Available: mean, median, stdev, variance, product, gcd, lcm, percentile_25, percentile_75, range"}
        return {"success": True, "operation": operation, "result": round(result, 6), "count": len(numbers)}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}

MATH_ADVANCED_DEF = ToolDefinition(
    name="math_advanced",
    description="Statistical and advanced math: mean, median, stdev, variance, product, gcd, lcm, percentile_25, percentile_75, range.",
    parameters=[
        ParameterSchema("operation", "string", "Operation name",
                        enum=["mean", "median", "stdev", "variance", "product", "gcd", "lcm", "percentile_25", "percentile_75", "range"]),
        ParameterSchema("numbers", "list", "List of numbers"),
    ],
    return_type="dict",
    return_description="{'success': bool, 'result': number}",
    category="math",
    version="1.0",
)


# ============================================================================
# Tool 10: Encoding Tools
# ============================================================================

def tool_encoding(text: str, operation: str) -> dict:
    """Text encoding/decoding operations."""
    try:
        if operation == "base64_encode":
            result = base64.b64encode(text.encode()).decode()
        elif operation == "base64_decode":
            result = base64.b64decode(text.encode()).decode()
        elif operation == "url_encode":
            result = urllib.parse.quote(text)
        elif operation == "url_decode":
            result = urllib.parse.unquote(text)
        elif operation == "md5":
            result = hashlib.md5(text.encode()).hexdigest()
        elif operation == "sha256":
            result = hashlib.sha256(text.encode()).hexdigest()
        elif operation == "char_codes":
            result = [ord(c) for c in text[:50]]  # limit to 50 chars
        elif operation == "from_char_codes":
            codes = json.loads(text)
            result = "".join(chr(c) for c in codes)
        else:
            return {"success": False, "error": f"Unknown operation '{operation}'. Available: base64_encode, base64_decode, url_encode, url_decode, md5, sha256, char_codes, from_char_codes"}
        return {"success": True, "operation": operation, "result": result}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}

ENCODING_DEF = ToolDefinition(
    name="encoding_tools",
    description="Encoding/decoding: base64_encode/decode, url_encode/decode, md5, sha256, char_codes, from_char_codes.",
    parameters=[
        ParameterSchema("text", "string", "Text to encode/decode"),
        ParameterSchema("operation", "string", "Encoding operation",
                        enum=["base64_encode", "base64_decode", "url_encode", "url_decode", "md5", "sha256", "char_codes", "from_char_codes"]),
    ],
    return_type="dict",
    return_description="{'success': bool, 'result': str}",
    category="utility",
    version="1.0",
)


# ============================================================================
# Registry Builder Extension
# ============================================================================

def register_production_tools(registry) -> None:
    """Register all 10 production tools into an existing ToolRegistry."""
    tools = [
        (CALCULATOR_DEF, tool_calculator),
        (STRING_UTILS_DEF, tool_string_utils),
        (LIST_OPS_DEF, tool_list_ops),
        (DICT_OPS_DEF, tool_dict_ops),
        (DATE_TIME_DEF, tool_date_time),
        (TEXT_STATS_DEF, tool_text_stats),
        (FORMAT_CONVERTER_DEF, tool_format_converter),
        (DATA_VALIDATOR_DEF, tool_data_validator),
        (MATH_ADVANCED_DEF, tool_math_advanced),
        (ENCODING_DEF, tool_encoding),
    ]
    for defn, fn in tools:
        registry.register(defn, fn)


# ============================================================================
# Test Suite
# ============================================================================

def run_production_tools_tests() -> None:
    """Run a comprehensive test suite against all 10 production tools."""
    from tool_registry import ToolRegistry, ToolTestHarness, ToolTestCase

    registry = ToolRegistry()
    register_production_tools(registry)
    harness = ToolTestHarness(registry)

    tests = [
        # Calculator
        ToolTestCase("calc_basic", "calculator", {"expression": "2 + 3"}, True, "5"),
        ToolTestCase("calc_sqrt", "calculator", {"expression": "sqrt(144)"}, True, "12"),
        ToolTestCase("calc_attack", "calculator", {"expression": "__import__('os')"}, False, None, "disallowed"),
        # String utils
        ToolTestCase("str_upper", "string_utils", {"text": "hello", "operation": "upper"}, True, "HELLO"),
        ToolTestCase("str_word_count", "string_utils", {"text": "one two three", "operation": "word_count"}, True, "3"),
        ToolTestCase("str_bad_op", "string_utils", {"text": "test", "operation": "explode"}, False, None, "not allowed"),
        # List ops
        ToolTestCase("list_sort", "list_ops", {"items": [3, 1, 2], "operation": "sort"}, True, "[1, 2, 3]"),
        ToolTestCase("list_frequencies", "list_ops", {"items": ["a", "a", "b"], "operation": "frequencies"}, True, "a"),
        # Dict ops
        ToolTestCase("dict_keys", "dict_ops", {"data": {"a": 1, "b": 2}, "operation": "keys"}, True, "['a', 'b']"),
        ToolTestCase("dict_get", "dict_ops", {"data": {"x": 42}, "operation": "get", "key": "x"}, True, "42"),
        ToolTestCase("dict_invert", "dict_ops", {"data": {"a": 1, "b": 2}, "operation": "invert"}, True, "{'1': 'a'"),
        # Date/time
        ToolTestCase("date_now", "date_time", {"operation": "now"}, True),
        ToolTestCase("date_parse", "date_time", {"operation": "parse", "date_str": "2024-06-15"}, True, "Saturday"),
        ToolTestCase("date_diff", "date_time", {"operation": "diff", "date_str": "2024-01-01", "fmt": "2024-12-31"}, True, "365"),
        # Text stats
        ToolTestCase("stats_basic", "text_stats", {"text": "Hello world. How are you?"}, True),
        # Format converter
        ToolTestCase("json_to_csv", "format_converter", {"data": '[{"name":"Alice","age":30}]', "from_format": "json", "to_format": "csv"}, True, "Alice"),
        ToolTestCase("json_to_md", "format_converter", {"data": '[{"name":"Alice","age":30}]', "from_format": "json", "to_format": "markdown_table"}, True, "Alice"),
        # Data validator
        ToolTestCase("valid_data", "data_validator", {"data": {"name": "Alice", "age": 30}, "rules": {"name": {"required": True, "type": "string", "min_length": 2}, "age": {"required": True, "type": "number", "min": 0, "max": 130}}}, True),
        ToolTestCase("invalid_age", "data_validator", {"data": {"name": "Alice", "age": 150}, "rules": {"name": {"required": True}, "age": {"max": 130}}}, True, "150"),
        # Advanced math
        ToolTestCase("math_mean", "math_advanced", {"operation": "mean", "numbers": [1, 2, 3, 4, 5]}, True, "3"),
        ToolTestCase("math_stdev", "math_advanced", {"operation": "stdev", "numbers": [10, 20, 30, 40, 50]}, True),
        # Encoding
        ToolTestCase("enc_b64", "encoding_tools", {"text": "hello", "operation": "base64_encode"}, True, "aGVsbG8"),
        ToolTestCase("enc_md5", "encoding_tools", {"text": "hello", "operation": "md5"}, True, "5d41402abc4b2a76b9719d911017c592"),
        ToolTestCase("enc_sha256", "encoding_tools", {"text": "hello", "operation": "sha256"}, True, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"),
    ]

    print("=" * 90)
    print("PRODUCTION TOOLS TEST SUITE (10 tools, 23 tests)")
    print("=" * 90)
    harness.run_suite(tests)

    # Registry stats
    stats = registry.get_stats()
    print(f"\nRegistry stats: {stats}")


if __name__ == "__main__":
    run_production_tools_tests()
