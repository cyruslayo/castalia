"""
Parser - Extract structured actions from raw LLM responses.

The LLM doesn't always return clean JSON. We need fallback strategies
to handle messy output while always returning a valid action dict.
"""

import json
import re
from typing import Dict, Any, Optional


def parse_response(text: str) -> Dict[str, str]:
    """
    Parse an LLM response and extract a structured action.
    
    Tries four strategies in order, falling back to a safe default.
    This guarantees the function always returns a usable dict, never crashes.
    
    Args:
        text: The raw string the LLM returned
        
    Returns:
        A dict with "action" key and a payload key ("thought" or "answer")
    """
    
    text = text or "".strip()  # Handle None defensively
    
    # Don't process empty strings
    if not text:
        return {"action": "think", "thought": ""}
    
    # ===== STRATEGY 1: Direct JSON parse =====
    result = try_direct_parse(text)
    if result is not None:
        return result
    
    # ===== STRATEGY 2: Extract from markdown code blocks =====
    result = try_markdown_extract(text)
    if result is not None:
        return result
    
    # ===== STRATEGY 3: Find first JSON object in the text =====
    result = try_brace_extract(text)
    if result is not None:
        return result
    
    # ===== STRATEGY 4: Look for answer-like keywords =====
    result = try_keyword_extract(text)
    if result is not None:
        return result
    
    # ===== FINAL FALLBACK: treat everything as a thought =====
    return {"action": "think", "thought": text[:500]}


# ====================================================================
# Strategy 1: Direct parse
# ====================================================================
def try_direct_parse(text: str) -> Optional[Dict[str, str]]:
    """
    Try to parse the entire string as JSON.
    
    This is the fastest path — if the LLM followed instructions perfectly,
    the whole response is valid JSON.
    
    Returns:
        Parsed dict if it contains an "action" key, None otherwise
    """
    try:
        data = json.loads(text)  # Try to convert string to dict
        if isinstance(data, dict) and "action" in data:  # Must be a dict with "action"
            return normalize_action(data)  # Clean up the keys
    except (json.JSONDecodeError, ValueError, TypeError):
        # Any parsing error means "not clean JSON", try next strategy
        pass
    return None  # Explicitly return None so the caller knows this failed


# ====================================================================
# Strategy 2: Markdown code block extraction
# ====================================================================
def try_markdown_extract(text: str) -> Optional[Dict[str, str]]:
    """
    Look for JSON inside markdown code blocks.
    
    LLMs often wrap output in:
        ```json
        {"action": "think", "thought": "..."}
        ```
    
    We use a regular expression to find and extract the inner JSON.
    """
    
    # Regex explanation:
    # ```          - literal opening backticks (3)
    # (?:json)?     - optional "json" label (non-capturing group)
    # \s*           - optional whitespace after the label
    # (\{.*?\})      - capture the JSON content (starts with {, ends with }, non-greedy)
    # \s*```         - optional whitespace then closing backticks
    pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    
    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL makes . match newlines too
    
    if match:
        json_string = match.group(1)  # Extract just the captured JSON part
        return try_parse_json(json_string)  # Try to parse it (reuses the same logic)
    
    return None  # No code block found


# ====================================================================
# Strategy 3: Brace extraction (find any {...} block)
# ====================================================================
def try_brace_extract(text: str) -> Optional[Dict[str, str]]:
    """
    Search for the first balanced {...} block anywhere in the text.
    
    This catches cases where the LLM writes:
        "Here is my response: {"action": "think", "thought": "..."}"
    
    The regex finds the outermost braces and everything inside.
    """
    
    # This regex finds the first { and matches until its closing }
    # It handles simple nesting (one level of inner braces)
    # [^{}]*        - match non-brace characters
    # (?:\{[^{}]*\} - or match a simple inner {...} group
    # [^{}]*)*      - repeat any combination
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    match = re.search(pattern, text)
    
    if match:
        json_string = match.group()  # The entire matched brace block
        return try_parse_json(json_string)
    
    return None


# ====================================================================
# Strategy 4: Keyword extraction
# ====================================================================
def try_keyword_extract(text: str) -> Optional[Dict[str, str]]:
    """
    If all JSON parsing fails, look for natural language patterns.
    
    Some LLMs just write:
        "The answer is 42."
        or
        "Final answer: the three states are solid, liquid, gas."
    
    We extract the text after these keywords.
    """
    
    # Look for patterns like "answer: ..." or "final answer: ..."
    # (?:...)       - non-capturing group for the keywords
    # [:\s]+         - colon or whitespace after the keyword
    # (.+)           - capture the actual answer text
    pattern = r'(?:answer|final answer|result)[:\s]+(.+)' 
    flags = re.IGNORECASE  # Make it case-insensitive (Answer, answer, ANSWER all match)
    
    match = re.search(pattern, text, flags)
    
    if match:
        answer_text = match.group(1).strip()  # Get just the captured answer part
        if answer_text:  # Only return if we actually captured something
            return {"action": "answer", "answer": answer_text}
    
    return None  # No keywords found


# ====================================================================
# Helper: Try to parse a JSON string and validate it
# ====================================================================
def try_parse_json(json_string: str) -> Optional[Dict[str, str]]:
    """
    Attempt to parse a string as JSON and validate it has an action.
    
    This is extracted so all strategies can reuse the same parsing + validation.
    """
    try:
        data = json.loads(json_string)
        if isinstance(data, dict) and "action" in data:
            return normalize_action(data)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


# ====================================================================
# Helper: Normalize the action dict
# ====================================================================
def normalize_action(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Ensure the action dict has the correct keys and values.
    
    The LLM might return extra fields or misspell keys.
    This enforces a consistent output format.
    
    CRITICAL: Also handles the common LLM pattern of nesting a real action
    inside a "think" or "answer". For example:
    {"action": "think", "thought": "{\"action\": \"use_tool\", ...}"}
    We detect this and extract the inner action.
    """
    
    action = data.get("action", "").strip().lower()  # Normalize to lowercase
    
    # Check for nested actions first - if the thought/answer contains a real action, use that
    # This handles the common LLM pattern of wrapping actions
    for field_key in ["thought", "answer", "reasoning", "content"]:
        if field_key in data:
            field_value = str(data[field_key]).strip()
            # Try to parse the field value as JSON that contains an action
            nested = _try_extract_nested_action(field_value)
            if nested:  # Found a real action nested inside
                return nested
    
    # No nested action found, process normally
    if action == "think":
        # Expect a "thought" key
        thought = str(data.get("thought", "")).strip()
        return {"action": "think", "thought": thought}
    
    elif action in ["answer", "final_answer", "result"]:
        # Expect an "answer" key
        answer = str(data.get("answer", data.get("final_answer", data.get("result", "")))).strip()
        result = {"action": "answer", "answer": answer}
        # Pass through the thought for ReAct (the reasoning behind the answer)
        if "thought" in data:
            result["thought"] = str(data["thought"]).strip()
        return result
    
    elif action in ["use_tool", "tool_use", "tool", "execute"]:
        # The LLM might put 'tool' inside 'params' or at the top level - handle both
        if "tool" in data:
            tool = str(data["tool"]).strip()
        elif "params" in data and isinstance(data["params"], dict) and "tool" in data["params"]:
            tool = str(data["params"]["tool"]).strip()
        else:
            tool = ""
        
        # Get params, excluding the 'tool' key if it's inside params
        params = data.get("params", {})
        if isinstance(params, dict) and "tool" in params:
            params = {k: v for k, v in params.items() if k != "tool"}
        
        # Pass through the thought for ReAct (the reasoning behind the action)
        result = {"action": "use_tool", "tool": tool, "params": params}
        if "thought" in data:
            result["thought"] = str(data["thought"]).strip()
        return result
    
    # If the action is something we don't recognize, treat it as a think
    # This prevents the agent from crashing on unexpected LLM output
    fallback_text = str(data.get("thought", data.get("answer", str(data)))).strip()
    return {"action": "think", "thought": fallback_text}


def _try_extract_nested_action(text: str) -> Optional[Dict[str, str]]:
    """
    Try to extract a real action from text that might be nested JSON.
    
    This handles the common LLM pattern where it wraps a use_tool or answer
    inside a think action. For example:
    Input: {"action": "use_tool", "tool": "python", "params": {...}}
    
    We only extract if the nested action is a "real" one (use_tool, answer),
    not another think (which would cause infinite unwrapping).
    
    Args:
        text: The field value that might contain nested JSON with an action
        
    Returns:
        A dict with the extracted action, or None if no valid nested action found
    """
    try:
        # Try to parse as JSON
        nested = json.loads(text)
        if isinstance(nested, dict) and "action" in nested:
            nested_action = nested["action"].strip().lower()
            
            # Only unwrap if it's a "real" action (not another think)
            # This prevents infinite recursion of think-within-think
            if nested_action in ["use_tool", "tool_use", "tool", "execute"]:
                # The LLM might put 'tool' inside 'params' or at the top level - handle both
                if "tool" in nested:
                    tool = str(nested["tool"]).strip()
                elif "params" in nested and isinstance(nested["params"], dict) and "tool" in nested["params"]:
                    tool = str(nested["params"]["tool"]).strip()
                else:
                    tool = ""
                
                # Get params, excluding the 'tool' key if it's inside params
                params = nested.get("params", {})
                if isinstance(params, dict) and "tool" in params:
                    # Remove 'tool' from params since we extracted it above
                    params = {k: v for k, v in params.items() if k != "tool"}
                
                result = {"action": "use_tool", "tool": tool, "params": params}
                # Pass through thought from the nested action when present.
                # Note: this helper receives only the nested text, so the outer
                # wrapper dict is intentionally unavailable here.
                if "thought" in nested:
                    result["thought"] = str(nested["thought"]).strip()
                return result
            
            elif nested_action in ["answer", "final_answer", "result"]:
                answer = str(nested.get("answer", nested.get("final_answer", nested.get("result", "")))).strip()
                return {"action": "answer", "answer": answer}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    return None


# ====================================================================
# Test the parser
# ====================================================================
if __name__ == "__main__":
    test_cases = [
        # (input, expected description)
        ('{"action": "think", "thought": "Let me break this down."}', 
         "Clean JSON - think action"),
        
        ('{"action": "answer", "answer": "The result is 42."}',
         "Clean JSON - answer action"),
        
        ('```json\n{"action": "think", "thought": "Considering the options..."}\n```',
         "JSON in markdown code block"),
        
        ('Here is my response: {"action": "answer", "answer": "42"}',
         "JSON embedded in natural language"),
        
        ('The answer is 42.',
         "Natural language with 'answer' keyword"),
        
        ('Final answer: the three states of matter are solid, liquid, and gas.',
         "'Final answer' keyword phrase"),
        
        ('Some random text without any structure or JSON',
         "Fallback to treating as a thought"),
        
        ('',
         "Empty string edge case"),
        
        # Nested action tests - the LLM wraps a real action inside a think/answer
        ('{"action": "think", "thought": "{\\"action\\": \\"use_tool\\", \\"tool\\": \\"python\\", \\"params\\": {\\"code\\": \\"print(42)\\"}}"}',
         "Nested: use_tool wrapped inside think"),
        
        ('{"action": "think", "thought": "{\\"action\\": \\"answer\\", \\"answer\\": \\"42\\"}"}',
         "Nested: answer wrapped inside think"),
        
        ('{"action": "use_tool", "tool": "read_file", "params": {"filepath": "data.txt"}}',
         "Direct use_tool action (no nesting)"),
    ]
    
    print("Testing the parser...\n" + "=" * 60)
    
    for test_input, description in test_cases:
        result = parse_response(test_input)
        action = result["action"]
        
        # Handle different action types for display
        if action == "think":
            payload_key = "thought"
            payload = result[payload_key]
        elif action == "answer":
            payload_key = "answer"
            payload = result[payload_key]
        elif action == "use_tool":
            payload_key = "tool"
            payload = f"tool={result['tool']}"
        else:
            payload_key = "?"
            payload = str(result)
        
        # Truncate long payloads for readability
        if len(payload) > 60:
            payload_display = payload[:60] + "..."
        else:
            payload_display = payload
        
        print(f"Test: {description}")
        print(f"  Input:  {test_input[:50]}{'...' if len(test_input) > 50 else ''}")
        print(f"  Output: action={action}, {payload_key}=\"{payload_display}\"")
        print()
    
    print("All tests complete.")
