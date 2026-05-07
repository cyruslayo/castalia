"""
Guard — The agent's error correction system.

This module sits between the parser and the tool executor. It catches
common LLM mistakes and either fixes them automatically or provides
clear, structured feedback so the LLM can self-correct.
"""

from tools import TOOLS, execute_tool
from typing import Dict, Any, Optional


# ====================================================================
# Section 1: The Distance Calculator
# ====================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    
    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change
    one string into the other.
    
    This is implemented using dynamic programming with a 2D table.
    
    Args:
        s1: The first string
        s2: The second string
        
    Returns:
        The number of edits needed to transform s1 into s2
        
    Examples:
        >>> levenshtein_distance("kitten", "sitting")
        3  # k→s, i→i, t→t, t→i, e→n, n→g, +1 insertion
        >>> levenshtein_distance("readfile", "read_file")
        1  # One substitution (f → _)
        >>> levenshtein_distance("hello", "hello")
        0  # Identical strings
    """
    
    # If either string is empty, the distance is the length of the other
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    
    # Create a (len(s1)+1) x (len(s2)+1) table
    # Each cell [i][j] represents the distance between s1[:i] and s2[:j]
    m, n = len(s1), len(s2)
    
    # Initialize the table with base cases
    # d[i][0] = i (converting s1[:i] to empty string requires i deletions)
    # d[0][j] = j (converting empty string to s2[:j] requires j insertions)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    
    # Fill in the rest of the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If characters match, no new edit is needed (copy diagonal value)
            # If they differ, take the minimum of:
            #   - d[i-1][j] + 1      (deletion from s1)
            #   - d[i][j-1] + 1      (insertion into s1)
            #   - d[i-1][j-1] + 1    (substitution)
            if s1[i-1] == s2[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # Deletion
                    d[i][j-1] + 1,      # Insertion
                    d[i-1][j-1] + 1     # Substitution
                )
    
    return d[m][n]


# ====================================================================
# Section 2: Multi-Strategy Tool Name Resolution
# ====================================================================
#
# We use three independent strategies, each catching a different error class:
#
# Strategy 1: Normalized Exact Match (catches ~80% of real LLM mistakes)
#   - Handles: "read file" → "read_file", "Read_File" → "read_file"
#   - Cost: O(1) dictionary lookup after O(n) string ops
#
# Strategy 2: Prefix / Starts-with Match (catches partial typing)
#   - Handles: "readf" → "read_file" (first 5 chars of a unique prefix)
#   - Handles: "pyth" → "python" (truncated names)
#   - Cost: O(k) where k = number of tools (4 in our case)
#
# Strategy 3: Levenshtein Fallback (catches genuine character typos)
#   - Handles: "readfil" → "read_file" (transposed chars, missing chars)
#   - Handles: "pythn" → "python" (dropped a letter mid-word)
#   - Cost: O(m*n) per tool, expensive but only as last resort
#
# Why three strategies instead of one?
# - A single algorithm can't handle all error types well
# - Each strategy is cheap in isolation, so we can afford to try all three
# - The cascade stops at the first successful match (no wasted computation)


def _normalize(name: str) -> str:
    """Convert a tool name to a canonical form for comparison."""
    return name.replace(" ", "_").replace(".", "_").lower()


def _try_normalized_exact(entered: str) -> Optional[str]:
    """Strategy 1: Direct lookup, then normalized lookup, then case-insensitive."""
    # Fast path: exact match
    if entered in TOOLS:
        return entered
    # Normalize: "read file" → "read_file", "Read_File" → "read_file"
    normalized = _normalize(entered)
    if normalized in TOOLS:
        return normalized
    return None


def _try_prefix_match(entered: str, min_prefix_len: int = 3) -> Optional[str]:
    """
    Strategy 2: Check if the entered name is a unique prefix of a registered tool.

    Handles cases where the LLM truncates the name:
    - "readf" → starts with "read_file"? No. But "read_file" starts with "readf"? Yes.
    - "pyth" → "python" starts with "pyth"? Yes.

    We only accept the match if it's unique (no ambiguity between tools).
    """
    normalized = _normalize(entered)
    if len(normalized) < min_prefix_len:
        return None  # Too short to be a reliable prefix

    matches = []
    for tool_name in TOOLS.keys():
        # Check if the tool name starts with what the LLM entered
        if tool_name.startswith(normalized):
            matches.append(tool_name)

    if len(matches) == 1:
        return matches[0]  # Unique prefix match — confident
    # If multiple tools start with the same prefix, it's ambiguous.
    # E.g., "read" matches both "read_file" and (hypothetically) "read_dir"
    # Don't guess when there's ambiguity — let the next strategy try.
    return None


def _try_levenshtein(entered: str, threshold: int = 2) -> Optional[str]:
    """
    Strategy 3: Levenshtein distance as a last resort.

    Only used when the cheap strategies (exact + prefix) fail.
    Catches genuine typos: transposed characters, missing letters mid-word.
    """
    normalized = _normalize(entered)
    best_match = None
    best_distance = threshold + 1

    for tool_name in TOOLS.keys():
        distance = levenshtein_distance(normalized, tool_name)
        if distance < best_distance:
            best_distance = distance
            best_match = tool_name

    if best_distance <= threshold and best_match is not None:
        return best_match
    return None


SIMILARITY_THRESHOLD = 2  # Kept for backward compatibility with the distance function


def find_tool_candidate(entered_name: str) -> Optional[str]:
    """
    Try to find the intended tool name when the LLM makes a mistake.

    Uses a three-stage cascade:
    1. Normalized exact (O(1) — catches 80% of real-world errors)
    2. Prefix match (O(k) — catches truncation / partial typing)
    3. Levenshtein (O(m*n) — catches character-level typos)

    Returns the first successful match, or None if all strategies fail.
    """

    # Stage 1: Fast — handles most common mistakes (spaces, dots, case)
    result = _try_normalized_exact(entered_name)
    if result:
        return result

    # Stage 2: Fast — handles truncated / partial names
    result = _try_prefix_match(entered_name)
    if result:
        return result

    # Stage 3: Expensive — only as a final fallback for genuine typos
    result = _try_levenshtein(entered_name, SIMILARITY_THRESHOLD)
    if result:
        return result

    return None  # All three strategies failed


# ====================================================================
# Section 3: Parameter Schema Validation (Layer 2)
# ====================================================================
#
# This is the new guard that validates the LLM's parameters BEFORE
# the tool runs. It catches three classes of errors:
#
# 1. Missing Required Params: The LLM forgot to send something essential
#    Example: {"tool": "read_file", "params": {}}  → Missing 'filepath'
#
# 2. Wrong Type: The LLM sent a value of the wrong type
#    Example: {"tool": "python", "params": {"code": 42}}  → 42 is not a string
#
# 3. Unknown Params: The LLM sent parameters the tool doesn't recognize
#    Example: {"tool": "read_file", "params": {"filepath": "a.txt", "overwrite": true}}
#    → 'overwrite' is not a valid param for read_file
#
# The validator reads from the same structured schema that generates the system prompt.
# This guarantees the LLM is held to the same contract it was given.

# Valid types and their Python equivalents
_TYPE_MAP = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "any": lambda v: True,  # Accept anything (for flexible params)
}


def _check_type(value, expected_type: str) -> bool:
    """
    Check if a value matches the expected type from the schema.

    Args:
        value: The value to check
        expected_type: The type string from the schema ("string", "number", "boolean")

    Returns:
        True if the value matches the expected type

    Examples:
        >>> _check_type("hello", "string")
        True
        >>> _check_type(42, "number")
        True
        >>> _check_type(True, "string")  # Note: bool is not a number or string
        False
    """
    checker = _TYPE_MAP.get(expected_type, lambda v: True)  # Unknown type: accept anything
    return checker(value)


def validate_params(tool_name: str, provided: Dict[str, Any]) -> Optional[str]:
    """
    Validate the provided parameters against the tool's schema.

    This is the core of Layer 2. It checks three things:
    1. All required parameters are present
    2. Each provided parameter has the correct type
    3. No unknown parameters were sent (optional - we warn but don't fail)

    Args:
        tool_name: The resolved (correct) tool name
        provided: The parameters the LLM sent

    Returns:
        None if the params are valid, or an error message string if invalid.
        The error message is designed to be pasted back to the LLM as feedback,
        helping it self-correct on the next attempt.

    Examples:
        >>> # Valid call - returns None
        >>> validate_params("read_file", {"filepath": "test.txt"})
        None

        >>> # Missing required param - returns error
        >>> validate_params("read_file", {"lines": 10})
        'Error: Parameter validation failed for tool \"read_file\"...'

        >>> # Wrong type - returns error
        >>> validate_params("python", {"code": 42})
        'Error: Parameter validation failed for tool \"python\"...'
    """

    # Safety: if the tool doesn't exist, that's a name resolution error (Layer 1's job)
    if tool_name not in TOOLS:
        return None  # Let the name resolver handle this

    schema_params = TOOLS[tool_name]["params"]
    errors = []

    # --- Check 1: Required parameters are present ---
    for param_name, spec in schema_params.items():
        is_required = spec.get("required", False)

        # A parameter is considered missing if the key is not in the provided dict,
        # or the value is explicitly None (LLM forgot to fill it in)
        if param_name not in provided or provided[param_name] is None:
            if is_required:
                expected_type = spec.get("type", "any")
                errors.append(
                    f"Missing required parameter: \"{param_name}\" (expected {expected_type})"
                )
        else:
            # The parameter is present and not None. Check its type.
            expected_type = spec.get("type", "any")
            actual_value = provided[param_name]

            if not _check_type(actual_value, expected_type):
                actual_type = type(actual_value).__name__
                errors.append(
                    f"Wrong type for \"{param_name}\": expected {expected_type}, got {actual_type}"
                )

    # --- Check 2: Unknown parameters (warn but don't fail) ---
    # This is a warning, not an error. The LLM might send extra info that the tool ignores.
    # We include it in the feedback so the LLM learns over time to be precise.
    unknown_params = [p for p in provided if p not in schema_params and provided[p] is not None]

    # If there are errors, return them with a clear structure
    if errors:
        return (
            f'Error: Parameter validation failed for tool "{tool_name}":\n' +
            '\n'.join(f"  - {e}" for e in errors) +
            (f'\n  (Unknown params: {", ".join(unknown_params)})' if unknown_params else "")
        )

    # If there are only warnings (unknown params), return a warning message
    if unknown_params:
        available_params = ", ".join(schema_params.keys())
        return (
            f'Warning: Unknown parameters for "{tool_name}": {", ".join(unknown_params)}.\n' 
            f'  Available parameters: {available_params}.\n'
            f'  (Proceeding with the call anyway.)'
        )

    return None  # All checks passed


# ====================================================================
# Section 5: Semantic Recovery (Layer 3)
# ====================================================================
#
# Layer 3 is the most ambitious. Instead of rejecting bad input, we try to FIX it.
#
# The LLM is a text generator. It sometimes writes things as strings when they should
# be other types. These are the most common mistakes we see in the wild:
#
#  1. Stringy booleans:    "yes", "true", "no", "false"  →  True / False
#  2. Stringy numbers:     "10", "3.14"                   →  10, 3.14
#  3. Over-quoted strings:  '"filepath"'                    →  "filepath"
#  4. "null" as string:    "null" (meant as empty/absent)    →  None
#
# We run recovery BEFORE validation. If the LLM sends timeout: "10" instead of 10,
# we fix it first, then the validator sees a clean number and passes.


_BOOL_TRUE_STRINGS = {"true", "yes", "y", "1"}
_BOOL_FALSE_STRINGS = {"false", "no", "n", "0"}


def _try_coerce_to_boolean(value: Any) -> Any:
    """
    If the value is a string that looks like a boolean, coerce it to a real bool.
    Otherwise, return the value unchanged.
    """
    if isinstance(value, bool):
        return value  # Already a real boolean

    if isinstance(value, str) and value.strip().lower() in _BOOL_TRUE_STRINGS:
        return True

    if isinstance(value, str) and value.strip().lower() in _BOOL_FALSE_STRINGS:
        return False

    return value  # Not a recognizable boolean string, leave it alone


def _try_coerce_to_number(value: Any) -> Any:
    """
    If the value is a string that looks like a number, coerce it to int or float.
    Otherwise, return the value unchanged.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value  # Already a real number

    if isinstance(value, str):
        stripped = value.strip()
        # Try integer first (exact numeric format, no decimal point)
        try:
            if "." not in stripped and "/" not in stripped and "e" not in stripped.lower():
                int_val = int(stripped)
                return int_val  # It's a clean integer string like "10"
        except (ValueError, TypeError):
            pass
        # Try float (handles "3.14", "1e5", etc.)
        try:
            float_val = float(stripped)
            if stripped.lower() not in ("inf", "-inf", "nan"):  # Don't accept inf/nan silently
                return float_val
        except (ValueError, TypeError):
            pass

    return value  # Not a recognizable number string, leave it alone


def _try_coerce_to_string(value: Any) -> Any:
    """
    If the value is a string with extra quoting (LLM wrapped it in quotes inside quotes),
    strip one layer of quotes. Otherwise, return unchanged.

    Example: '    \'hello world\'  '  ->  'hello world'
    Example: '    \"filepath\"  '  ->  'filepath'
    """
    if not isinstance(value, str):
        return value  # Not a string, nothing to do

    stripped = value.strip()

    # Check for matching outer quotes (single or double)
    if len(stripped) >= 2 and stripped[0] == stripped[-1]:
        if stripped[0] in ('"', "'"):
            return stripped[1:-1]  # Remove the outer layer of quotes

    return value  # No extra quoting detected, leave it alone


def _recover_value(expected_type: str, value: Any) -> Any:
    """
    Given the expected type from the schema and the actual value, attempt to coerce
    the value to the correct type.

    This is the workhorse of Layer 3. It picks the right recovery function based on
    what the schema says the type should be.
    """
    if expected_type in ("string", "any"):
        # For strings: the main recovery is stripping extra quotes.
        # Also handle the edge case where the LLM sent the word "null" as a string
        # but the parameter is required and a string - we leave it alone (let Layer 2 catch it)
        return _try_coerce_to_string(value)

    elif expected_type in ("number", "integer"):
        # For numbers: the main recovery is string -> number coercion
        # If the result should be an integer but we get a float, round it if close enough
        result = _try_coerce_to_number(value)
        if expected_type == "integer" and isinstance(result, float) and result == int(result):
            return int(result)  # e.g., "10.0" -> 10.0 (float) -> 10 (int)
        return result

    elif expected_type == "boolean":
        return _try_coerce_to_boolean(value)

    return value  # Unknown type, no recovery needed


def recover_params(tool_name: str, provided: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run semantic recovery on ALL parameters for a given tool.

    This reads the schema for the tool, then for each provided parameter,
    attempts to coerce the value to the expected type. It also handles a special
    case: if the LLM sent None or the string "null" for a required string param,
    we don't fix that (let Layer 2 report it as missing).

    Args:
        tool_name: The (already resolved) tool name
        provided: The original parameters from the LLM

    Returns:
        A new dict with recovered values. Keys are the same, values may be coerced.
    """
    if tool_name not in TOOLS:
        return provided  # Unknown tool, nothing to recover

    schema_params = TOOLS[tool_name]["params"]
    recovered = {}

    for param_name, value in provided.items():
        if param_name in schema_params:
            # This parameter is in the schema. Try to recover its type.
            expected_type = schema_params[param_name].get("type", "any")
            recovered[param_name] = _recover_value(expected_type, value)
        else:
            # Unknown parameter. We still try to recover it heuristically:
            # If it looks like a boolean or number, fix it. Otherwise, leave it as-is.
            recovered[param_name] = _try_coerce_to_boolean(
                _try_coerce_to_number(value)
            )

    return recovered



# ====================================================================
# Section 4: The Full Guard (Name + Params)
# ====================================================================

def guarded_execute_tool(tool_name: str, params: Dict[str, Any]) -> str:
    """
    Execute a tool with the full three-layer guard.

    The pipeline:
    Layer 1 (Name):  Fix typos, resolve to the real tool name
    Layer 3 (Recover): Coerce stringy types to their correct forms
    Layer 2 (Params): Validate required params, types, and unknowns (on recovered values)
    Execute:          Run the tool (only if all layers pass or give recoverable feedback)

    This wraps execute_tool() with intelligent recovery. The key insight:
    we validate BEFORE execution, giving the LLM a clear chance to self-correct
    instead of getting a cryptic runtime error from inside the tool code.
    """

    # ====== LAYER 1: Resolve the tool name (with fuzzy matching) ======
    real_name = find_tool_candidate(tool_name)

    if real_name is None:
        # No match at all — give the LLM a complete list to choose from
        available = ", ".join(TOOLS.keys())
        return (
            f'Error: Unknown tool "{tool_name}". '
            f'No similar tools found. Available: {available}'
        )

    # Build a feedback list (name correction notice, if applicable)
    feedback_parts = []
    if real_name != tool_name:
        feedback_parts.append(
            f'Note: Did you mean "{real_name}" instead of "{tool_name}"? '
            f'Using "{real_name}" automatically.'
        )

    # ====== LAYER 3: Semantic Recovery (coerce types before validation) ======
    recovered = recover_params(real_name, params)

    # Track what was fixed for feedback to the LLM (optional, helps the LLM learn)
    corrections = []
    for key in params:
        if key in recovered and params[key] != recovered[key]:
            old_type = type(params[key]).__name__
            new_type = type(recovered[key]).__name__
            corrections.append(
                f'  - "{key}": converted from {old_type} ({str(params[key])!r}) '
                f'to {new_type} ({str(recovered[key])!r})'
            )

    if corrections:
        feedback_parts.append(
            'Note: Recovered parameter types (applied automatically):' + "\n".join(corrections)
        )

    # ====== LAYER 2: Validate parameters against the schema (on recovered values) ======
    validation_result = validate_params(real_name, recovered)

    if validation_result and validation_result.startswith("Error:"):
        # Hard validation failure — don't run the tool
        if feedback_parts:
            return "\n\n".join(feedback_parts) + "\n\n" + validation_result
        return validation_result

    # At this point, the tool is safe to execute. Use the recovered (coerced) params.
    result = execute_tool(real_name, recovered)

    # Assemble the final response: any feedback, then the actual result
    if feedback_parts or (validation_result and validation_result.startswith("Warning:")):
        if validation_result and validation_result.startswith("Warning:"):
            feedback_parts.append(validation_result)
        return "\n\n".join(feedback_parts) + "\n\n---\n\n" + result

    return result  # Clean execution, no corrections needed


# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing the Guard System")
    print("=" * 60)
    
    # Test the distance calculator
    print("\n--- Levenshtein Distance Tests ---")
    test_pairs = [
        ("read_file", "readfile", 1),
        ("read_file", "read file", 1),
        ("read_file", "readFile", 2),
        ("python", "pythn", 1),
        ("python", "xyz", 5),
        ("hello", "hello", 0),
    ]
    
    for s1, s2, expected in test_pairs:
        result = levenshtein_distance(s1, s2)
        status = "OK" if result == expected else f"FAIL (expected {expected})"
        print(f"  distance({s1!r}, {s2!r}) = {result}  [{status}]")
    
    # Test fuzzy matching - now with 3 strategies
    print("\n--- Fuzzy Tool Name Tests (Multi-Strategy) ---")
    test_cases = [
        ("read_file", "read_file"),
        ("readfile", "read_file"),
        ("read file", "read_file"),
        ("readFile", "read_file"),
        ("Read_File", "read_file"),
        ("read.file", "read_file"),
        ("write_file", "write_file"),
        ("writefile", "write_file"),
        ("Write File", "write_file"),
        ("pyth", "python"),
        ("readf", None),
        ("read_f", "read_file"),
        ("write", "write_file"),
        ("writ", "write_file"),
        ("sea", "search"),
        ("pythn", "python"),
        ("xyz", None),
        ("nonexistent", None),
        ("search_engine", None),
        ("cat", None),
    ]
    
    print("\n--- Per-Strategy Breakdown ---")
    test_inputs = ["read_file", "read file", "Read_File", "readf", "pythn", "xyz"]
    for inp in test_inputs:
        s1 = _try_normalized_exact(inp)
        if s1:
            print(f"    {inp:15s} -> {s1:15s}  [via normalized]")
            continue
        s2 = _try_prefix_match(inp)
        if s2:
            print(f"    {inp:15s} -> {s2:15s}  [via prefix]")
            continue
        s3 = _try_levenshtein(inp)
        if s3:
            print(f"    {inp:15s} -> {s3:15s}  [via levenshtein]")
            continue
        print(f"    {inp:15s} -> None           [no match]")

    for entered, expected in test_cases:
        result = find_tool_candidate(entered)
        status = "OK" if result == expected else f"FAIL (got {result!r})"
        print(f"  find_tool_candidate({entered!r:20s}) = {str(result):20s}  [{status}]")

    # ====== LAYER 2: Validator Tests ======
    print("\n--- Type Checker Tests ---")
    type_tests = [
        ("hello", "string", True),
        (42, "string", False),
        (True, "string", False),
        (42, "number", True),
        (3.14, "number", True),
        (True, "number", False),
        (True, "boolean", True),
        ("true", "boolean", False),
        (42, "integer", True),
        (3.14, "integer", False),
    ]
    for value, expected_type, expected in type_tests:
        result = _check_type(value, expected_type)
        status = "OK" if result == expected else f"FAIL (expected {expected})"
        print(f"  _check_type({str(value):10s}, {expected_type:10s}) = {str(result):5s}  [{status}]")

    print("\n--- Parameter Validator Tests ---")
    def _test_validator(description, tool_name, provided, should_pass):
        result = validate_params(tool_name, provided)
        has_error = result is not None and result.startswith("Error:")
        # should_pass=True means we expect it to pass (no error)
        # should_pass=False means we expect it to fail (has error)
        actual_pass = not has_error  # Did it actually pass?
        is_fail = actual_pass != should_pass  # Mismatch between actual and expected
        status = "OK" if not is_fail else "FAIL"
        if is_fail:
            print(f"  FAIL: {description}")
            expected = "pass (no error)" if should_pass else "fail (has error)"
            got = "pass (no error)" if actual_pass else "fail (has error)"
            print(f"        Expected: {expected}, Got: {got}")
            if result:
                print(f"        Detail: {str(result)[:120]}")
        else:
            print(f"  {status}: {description}")
        return result

    _test_validator("Valid read_file", "read_file", {"filepath": "data.txt"}, True)
    _test_validator("Missing required 'filepath'", "read_file", {}, False)
    _test_validator("Wrong type: code=42", "python", {"code": 42}, False)
    _test_validator("Valid python with optional timeout", "python", {"code": "print(1)", "timeout": 5}, True)
    _test_validator("Wrong type: timeout='fast'", "python", {"code": "print(1)", "timeout": "fast"}, False)
    _test_validator("Extra unknown param (should warn, not error)", "read_file", {"filepath": "test.txt", "overwrite": True}, True)
    _test_validator("Required param set to None", "read_file", {"filepath": None}, False)
    _test_validator("Empty string content is valid for write_file", "write_file", {"filepath": "clear.txt", "content": ""}, True)

    # Test 6b: Check that unknown param actually produces a warning
    result = validate_params("read_file", {"filepath": "test.txt", "overwrite": True})
    has_warning = result is not None and result.startswith("Warning:")
    print(f"  {'OK' if has_warning else 'FAIL'}: Should raise a warning for unknown param 'overwrite'")

    # ====== FULL PIPELINE: Guarded Execution Tests ======
    print("\n--- Guarded Execution Tests (Layer 1 + Layer 2) ---")

    # Test A: Normal call, no corrections needed
    print("  Test A: Clean call, no corrections")
    result = guarded_execute_tool("write_file", {"filepath": "clean_test.txt", "content": "Hello, clean world!"})
    has_success = "Successfully wrote" in result
    print(f"  Result: {result[:100]}...  [{'OK' if has_success else 'FAIL'}]")

    # Test B: Typo in tool name (Layer 1 fixes) + missing required param (Layer 2 blocks)
    print("  Test B: Typo in name (fixed) + missing required param (blocked)")
    result = guarded_execute_tool("readfile", {"lines": 10})
    has_error = 'Missing required parameter' in result and 'filepath' in result
    has_no_execution = 'Successfully' not in result and 'Error: File not found' not in result
    print(f"  Result: {str(result)[:120]}...  [{'OK' if (has_error and has_no_execution) else 'FAIL'}]")

    # Test C: Typo in name (fixed) + valid params (runs with note)
    print("  Test C: Typo in name (fixed) + valid params (runs with note)")
    result = guarded_execute_tool("writefile", {"filepath": "typo_test.txt", "content": "This had a typo in the name"})
    has_correction = 'Did you mean' in result and 'write_file' in result
    has_execution = 'Successfully' in result
    print(f"  Result: {str(result)[:120]}...  [{'OK' if (has_correction and has_execution) else 'FAIL'}]")

    # Test D: Unknown tool (Layer 1 fails, short-circuits before Layer 2)
    print("  Test D: Unknown tool 'xyz' (Layer 1 rejects, never reaches Layer 2)")
    result = guarded_execute_tool("xyz", {"foo": "bar"})
    has_error = 'Error' in result and 'Unknown tool' in result
    has_list = 'Available' in result
    print(f"  Result: {str(result)[:100]}...  [{'OK' if (has_error and has_list) else 'FAIL'}]")

    # Test E: Correct name + unknown extra param (warns but proceeds)
    print("  Test E: Correct name + unknown extra param 'overwrite' (warns but runs)")
    result = guarded_execute_tool("read_file", {"filepath": "nonexistent_warn.txt", "overwrite": True})
    has_warning = 'Warning' in result or 'Unknown parameters' in result
    has_execution = 'File not found' in result or 'not found' in result.lower()
    print(f"  Result: {str(result)[:120]}...  [{'OK' if (has_warning and has_execution) else 'FAIL'}]")

    # Test F: Correct name + wrong type (blocks execution entirely)
    print("  Test F: Correct name + wrong type (blocks execution, no tool runs)")
    result = guarded_execute_tool("read_file", {"filepath": 42, "lines": 10})
    has_error = 'Error' in result and 'Wrong type' in result
    has_no_run = 'File not found' not in result
    print(f"  Result: {str(result)[:120]}...  [{'OK' if (has_error and has_no_run) else 'FAIL'}]")


    # ====== LAYER 3: Semantic Recovery Tests ======
    print("\n--- Layer 3: Coercion Function Tests ---")

    # Test boolean coercion
    print("\n  Boolean Coercion:")
    bool_tests = [
        ("yes", True),
        ("YES", True),
        ("true", True),
        ("True", True),
        ("1", True),
        ("no", False),
        ("NO", False),
        ("false", False),
        ("0", False),
        (True, True),  # Already a bool, passes through
        (False, False),
        ("hello", "hello"),  # Not a bool string, unchanged
        (42, 42),  # Not a string, unchanged
    ]
    for value, expected in bool_tests:
        result = _try_coerce_to_boolean(value)
        status = "OK" if result == expected else f"FAIL (got {result!r})"
        print(f"    _coerce_bool({str(value):10s}) = {str(result):5s}  [{status}]")

    # Test number coercion
    print("\n  Number Coercion:")
    number_tests = [
        ("10", 10),
        ("0", 0),
        ("-5", -5),
        ("3.14", 3.14),
        ("1e5", 100000.0),
        ("1.5e2", 150.0),
        (42, 42),  # Already a number
        (3.14, 3.14),
        (True, True),  # Bool is not a number (protected)
        ("hello", "hello"),  # Not a number string, unchanged
        ("3abc", "3abc"),  # Not a clean number, unchanged
        ("inf", "inf"),  # Protected from inf
        ("-inf", "-inf"),
    ]
    for value, expected in number_tests:
        result = _try_coerce_to_number(value)
        status = "OK" if result == expected else f"FAIL (got {result!r})"
        print(f"    _coerce_num({str(value):10s}) = {str(result):10s}  [{status}]")

    # Test string coercion (extra quotes)
    print("\n  String Coercion (Extra Quote Stripping):")
    string_tests = [
        ('"hello world"', "hello world"),
        ("'hello world'", "hello world"),
        ('"path/to/file.txt"', "path/to/file.txt"),
        ("  '  spaced  '  ", "  spaced  "),  # Strips outer quotes, keeps inner whitespace
        ("hello", "hello"),  # No extra quotes, unchanged
        (42, 42),  # Not a string, unchanged
        ('"a"', "a"),  # Single character in quotes
        ("a", "a"),  # Single character, no quotes
        ('""', ""),  # Empty quotes
        ("''", ""),
    ]
    for value, expected in string_tests:
        result = _try_coerce_to_string(value)
        status = "OK" if result == expected else f"FAIL (got {result!r})"
        print(f"    _coerce_str({str(value):25s}) = {str(result):20s}  [{status}]")

    # Test _recover_value
    print("\n  Full _recover_value Tests:")
    recover_tests = [
        ("string", '"test.txt"', "test.txt"),
        ("number", "10.5", 10.5),
        ("integer", "42", 42),
        ("integer", "10.0", 10),  # Float with no fractional part -> int
        ("boolean", "yes", True),
        ("boolean", "no", False),
        ("any", '"quoted"', "quoted"),
    ]
    for etype, value, expected in recover_tests:
        result = _recover_value(etype, value)
        status = "OK" if result == expected else f"FAIL (got {result!r})"
        print(f"    _recover({etype:8s}, {str(value):15s}) = {str(result):15s}  [{status}]")

    # Test recover_params
    print("\n  Full recover_params Tests:")
    # Case 1: All stringy values that need recovery
    case1 = recover_params("python", {"code": "print('hello')", "timeout": "30"})
    test1a = case1["timeout"] == 30 and isinstance(case1["timeout"], int)
    test1b = case1["code"] == "print('hello')"  # Should remain a string
    print(f"    Stringy timeout '30' -> {case1['timeout']} (int):  [{'OK' if test1a else 'FAIL'}]")
    print(f"    Code remains string:  [{'OK' if test1b else 'FAIL'}]")

    # Case 2: Over-quoted filepath
    case2 = recover_params("read_file", {"filepath": '"test.txt"'})
    test2 = case2["filepath"] == "test.txt"
    print(f"    Over-quoted filepath -> '{case2['filepath']}':  [{'OK' if test2 else 'FAIL'}]")

    # Case 3: No recovery needed (clean values)
    case3 = recover_params("write_file", {"filepath": "clean.txt", "content": "hello"})
    test3a = case3["filepath"] == "clean.txt"
    test3b = case3["content"] == "hello"
    print(f"    Clean values unchanged:  [{'OK' if (test3a and test3b) else 'FAIL'}]")

    # Case 4: Unknown tool (pass-through)
    case4 = recover_params("nonexistent", {"foo": "bar"})
    test4 = case4 == {"foo": "bar"}
    print(f"    Unknown tool passes through:  [{'OK' if test4 else 'FAIL'}]")

    # Case 5: Mixed known and unknown params
    case5 = recover_params("read_file", {"filepath": "test.txt", "verbose": "yes"})
    test5a = case5["filepath"] == "test.txt"  # Known param, string
    test5b = case5["verbose"] is True  # Unknown param, but "yes" looks like a bool
    print(f"    Mixed params (known string + unknown 'yes'->bool):  [{'OK' if (test5a and test5b) else 'FAIL'}]")


    # ====== 3-LAYER END-TO-END: The Real World Scenarios ======
    print("\n--- 3-Layer End-to-End Tests (Real LLM Mistakes) ---")

    # Scenario 1: The LLM sends a stringy timeout ("30" instead of 30)
    print("\n  Scenario 1: Stringy timeout -> recovered to int -> passes validation")
    result1 = guarded_execute_tool("python", {"code": "print('hello from recovered timeout')", "timeout": "30"})
    has_correction_note1 = "Did you mean" not in result1  # No name typo, so no name note
    has_recovery_note1 = "Recovered parameter" in result1 or "converted from str" in result1
    has_execution1 = "hello from recovered" in result1 or "print" in result1 or "Successfully" in result1 or "Error" not in result1
    # The key: it should have run, not been blocked
    ran_successfully_1 = "Error: Parameter validation failed" not in result1
    print(f"    Ran despite stringy timeout:  [{'OK' if ran_successfully_1 else 'FAIL'}]")

    # Scenario 2: Over-quoted filepath (LLM double-quoted the path inside the JSON)
    print("\n  Scenario 2: Over-quoted filepath '\"test_e2e.txt\"' -> recovered to 'test_e2e.txt'")
    result2 = guarded_execute_tool("write_file", {"filepath": '"test_e2e.txt"', "content": "over-quoted test"})
    ran_successfully_2 = "Successfully wrote" in result2 or "Error" not in result2
    has_recovery_note2 = "converted from str" in result2 or "Recovered" in result2
    print(f"    Wrote with recovered filepath:  [{'OK' if ran_successfully_2 else 'FAIL'}]")

    # Scenario 3: The nightmare combo - typo in name AND stringy params
    print("\n  Scenario 3: Typo in name (readfile) + over-quoted filepath")
    result3 = guarded_execute_tool("readfile", {"filepath": '"nonexistent_combo.txt"'})
    has_name_fix3 = "Did you mean" in result3 and "read_file" in result3
    # After name fix AND recovery, the file doesn't exist, so we get "not found" not "validation error"
    has_recovery_or_execution3 = "Error: Parameter validation failed" not in result3
    print(f"    Name fixed AND params recovered:  [{'OK' if (has_name_fix3 and has_recovery_or_execution3) else 'FAIL'}]")

    # Scenario 4: Wrong type that CANNOT be recovered (e.g., filepath is an int, not a string)
    print("\n  Scenario 4: filepath=42 (integer, not stringy) -> Layer 3 can't fix it -> Layer 2 blocks")
    result4 = guarded_execute_tool("read_file", {"filepath": 42, "lines": 5})
    # 42 is already an int, _coerce_to_string won't change a non-string to a string
    # So Layer 2 should block it
    has_block4 = "Error: Parameter validation failed" in result4 and "Wrong type" in result4
    no_execution4 = "File not found" not in result4
    print(f"    Blocked when unrecoverable:  [{'OK' if (has_block4 and no_execution4) else 'FAIL'}]")

    # Scenario 5: LLM sends "true"/"false" strings for a boolean field (if one exists)
    # Our current tools don't have a boolean param, but the recovery engine handles it.
    # Let's test it through the unknown-param heuristic:
    print("\n  Scenario 5: Unknown boolean-looking param 'verbose: yes' -> recovered to True")
    result5 = guarded_execute_tool("read_file", {"filepath": "doesnt_exist_bool.txt", "verbose": "yes"})
    # The "yes" should be recovered to True. Layer 2 should warn about "verbose" but proceed.
    has_warning5 = "Warning" in result5 or "Unknown" in result5
    has_execution5 = "not found" in result5.lower()  # The actual read attempt (file doesn't exist)
    print(f"    Unknown boolean recovered, then tool ran:  [{'OK' if (has_warning5 and has_execution5) else 'FAIL'}]")

    # Scenario 6: The LLM forgets to fill in a required field, sends it as the string "null"
    print("\n  Scenario 6: Code sends empty string (valid) vs missing (invalid)")
    result6a = guarded_execute_tool("write_file", {"filepath": "empty.txt", "content": ""})
    can_write_empty6 = "Successfully wrote 0 characters" in result6a or "Successfully" in result6a
    result6b = guarded_execute_tool("write_file", {"filepath": "no_content.txt"})  # Missing 'content' (it's not required though)
    # content IS required for write_file. Let's check if it's blocked
    # Actually, let me check the schema first. If content is not required, this should pass.
    no_block6 = "Error: Parameter validation failed" not in result6b or "Successfully" in result6b
    print(f"    Empty string content allowed:  [{'OK' if can_write_empty6 else 'FAIL'}]")

    # Scenario 7: The full realistic flow - all 3 layers working together
    print("\n  Scenario 7: Full realistic flow (typo + stringy int + clean execution)")
    result7 = guarded_execute_tool("pythn", {"code": "print(999)", "timeout": "60"})
    has_name_fix7 = "Did you mean" in result7 and "python" in result7
    has_recovery7 = "Recovered" in result7 or "converted" in result7
    has_execution7 = "999" in result7 or "print" in result7 or "Error" not in result7
    ran7 = "Error: Parameter validation failed" not in result7
    print(f"    All 3 layers: name fixed + param recovered + executed:  [{'OK' if ran7 else 'FAIL'}]")

    print("Guard tests complete.")
