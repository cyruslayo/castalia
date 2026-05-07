"""
Output sanitizer for production code execution.

Cleans, validates, and limits output from sandboxed code execution.
Prevents data leakage, output-based attacks, and resource exhaustion.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, Any


# ─── Configuration ───────────────────────────────────────────────

MAX_OUTPUT_LENGTH = 1_000_000       # 1MB output limit
MAX_SINGLE_LINE = 10_000            # Per-line limit (prevents massive single lines)
TRUNCATION_MARKER = "\n... [OUTPUT TRUNCATED]"

# Patterns that indicate sensitive data leakage
SENSITIVE_PATTERNS = [
    # API keys, tokens, secrets
    (r'(?:api[_-]?key|apikey)\s*[=:]\s*["\']?[A-Za-z0-9_\-]{16,}["\']?',
     'API_KEY_LEAK', 'Potential API key detected'),

    (r'(?:secret[_-]?key|secret)\s*[=:]\s*["\']?[A-Za-z0-9_\-]{16,}["\']?',
     'SECRET_LEAK', 'Potential secret key detected'),

    (r'(?:token|auth[_-]?token|access[_-]?token)\s*[=:]\s*["\']?[A-Za-z0-9_\-.]{20,}["\']?',
     'TOKEN_LEAK', 'Potential auth token detected'),

    # AWS credentials
    (r'AKIA[0-9A-Z]{16}',
     'AWS_ACCESS_KEY', 'AWS access key pattern detected'),

    # Private keys
    (r'-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----',
     'PRIVATE_KEY', 'Private key detected'),

    # Passwords
    (r'(?:password|passwd|pwd)\s*[=:]\s*["\']?[^\s"\']{8,}["\']?',
     'PASSWORD_LEAK', 'Potential password detected'),

    # Email addresses (PII concern)
    (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
     'EMAIL_PII', 'Email address detected (PII)'),

    # Credit card numbers (basic Luhn-range check)
    (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b',
     'CREDIT_CARD', 'Potential credit card number detected'),

    # Social Security Numbers
    (r'\b\d{3}-\d{2}-\d{4}\b',
     'SSN', 'Potential SSN detected'),

    # File paths (potential info leak)
    (r'(?:C:\\|/home/|/Users/|/etc/)[^\s"\']{5,}',
     'FILE_PATH', 'File path detected (potential info leak)'),
]


# ─── Data Classes ────────────────────────────────────────────────

@dataclass
class SanitizationReport:
    """Report of what was sanitized from output."""
    original_length: int
    sanitized_length: int
    was_truncated: bool
    issues_found: list = None
    redaction_count: int = 0

    def __post_init__(self):
        if self.issues_found is None:
            self.issues_found = []

    def has_issues(self) -> bool:
        return len(self.issues_found) > 0

    def summary(self) -> str:
        parts = [f"Output: {self.sanitized_length} chars"]
        if self.was_truncated:
            parts.append(f"TRUNCATED from {self.original_length}")
        if self.redaction_count > 0:
            parts.append(f"REDACTED {self.redaction_count} sensitive patterns")
        if self.has_issues():
            parts.append(f"ISSUES: {', '.join(i['type'] for i in self.issues_found)}")
        return " | ".join(parts)


@dataclass
class SanitizedOutput:
    """Clean output ready to return to the agent."""
    content: str
    content_type: str  # "text", "json", "error"
    report: SanitizationReport

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "content_type": self.content_type,
            "report": {
                "original_length": self.report.original_length,
                "sanitized_length": self.report.sanitized_length,
                "was_truncated": self.report.was_truncated,
                "redaction_count": self.report.redaction_count,
                "issues_found": self.report.issues_found,
            }
        }


# ─── Sanitization Engine ─────────────────────────────────────────

def sanitize_output(
    raw_output: str,
    max_length: int = MAX_OUTPUT_LENGTH,
    max_line_length: int = MAX_SINGLE_LINE,
    check_sensitive: bool = True,
) -> SanitizedOutput:
    """
    Sanitize output from code execution.

    Args:
        raw_output: Raw stdout/stderr from execution
        max_length: Maximum total output length
        max_line_length: Maximum length per individual line
        check_sensitive: Whether to scan for sensitive data patterns

    Returns:
        SanitizedOutput with cleaned content and audit report
    """
    report = SanitizationReport(
        original_length=len(raw_output),
        sanitized_length=0,
        was_truncated=False,
        issues_found=[],
        redaction_count=0,
    )

    # Step 1: Truncate if too long
    output = raw_output
    if len(output) > max_length:
        output = output[:max_length] + TRUNCATION_MARKER
        report.was_truncated = True

    # Step 2: Break massive single lines
    lines = output.split('\n')
    processed_lines = []
    for line in lines:
        if len(line) > max_line_length:
            processed_lines.append(line[:max_line_length] + "... [LINE TRUNCATED]")
        else:
            processed_lines.append(line)
    output = '\n'.join(processed_lines)

    # Step 3: Remove null bytes and other control characters
    output = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', output)

    # Step 4: Scan for sensitive patterns
    if check_sensitive:
        output, report = _scan_and_redact(output, report)

    # Step 5: Detect content type
    content_type = _detect_content_type(output)

    report.sanitized_length = len(output)

    return SanitizedOutput(
        content=output,
        content_type=content_type,
        report=report,
    )


def _scan_and_redact(output: str, report: SanitizationReport) -> tuple[str, SanitizationReport]:
    """Scan output for sensitive patterns and redact them."""
    for pattern, issue_type, description in SENSITIVE_PATTERNS:
        matches = list(re.finditer(pattern, output, re.IGNORECASE | re.MULTILINE))
        if matches:
            for match in matches:
                report.issues_found.append({
                    "type": issue_type,
                    "description": description,
                    "position": match.start(),
                })
            # Redact all matches
            output = re.sub(pattern, f"[REDACTED:{issue_type}]", output, flags=re.IGNORECASE | re.MULTILINE)
            report.redaction_count += len(matches)

    return output, report


def _detect_content_type(output: str) -> str:
    """Detect if output is JSON, text, or error."""
    stripped = output.strip()

    # Try JSON first
    if stripped.startswith('{') or stripped.startswith('['):
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            pass

    # Check for error indicators
    error_indicators = [
        "Traceback (most recent call last)",
        "Error:",
        "Exception:",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "NameError",
        "ImportError",
        "SyntaxError",
    ]
    for indicator in error_indicators:
        if indicator in output:
            return "error"

    # Default to text
    return "text"


# ─── Result Formatting for LLM ───────────────────────────────────

def format_for_llm(sanitized: SanitizedOutput, max_display: int = 2000) -> str:
    """
    Format sanitized output for LLM consumption.

    The LLM needs to understand:
    - What the code produced
    - Whether it was successful
    - Any truncation or redaction notices
    """
    parts = []

    # Add sanitization notices
    if sanitized.report.was_truncated:
        parts.append(
            f"[WARNING] Output was truncated: {sanitized.report.original_length} chars → "
            f"{sanitized.report.sanitized_length} chars (max {MAX_OUTPUT_LENGTH})"
        )

    if sanitized.report.redaction_count > 0:
        parts.append(
            f"[REDACTED] {sanitized.report.redaction_count} sensitive pattern(s) redacted"
        )

    if sanitized.report.has_issues():
        issue_types = set(i["type"] for i in sanitized.report.issues_found)
        parts.append(f"[SECURITY] Security notices: {', '.join(sorted(issue_types))}")

    # Add content type header
    type_headers = {
        "json": "[JSON] JSON Output:",
        "error": "[ERROR] Error Output:",
        "text": "[TEXT] Text Output:",
    }
    parts.append(type_headers.get(sanitized.content_type, "[OUTPUT] Output:"))

    # Add content (with display limit)
    content = sanitized.content
    if len(content) > max_display:
        content = content[:max_display] + "\n... [DISPLAY TRUNCATED]"

    parts.append(content)

    return "\n".join(parts)


def format_error_for_llm(error: str, exit_code: int, max_display: int = 1500) -> str:
    """
    Format execution error for LLM self-correction.

    The LLM needs:
    - The exact error message
    - Exit code
    - Helpful guidance on what went wrong
    """
    parts = [f"[ERROR] Execution failed (exit code: {exit_code})"]

    # Truncate very long tracebacks
    if len(error) > max_display:
        # Keep the last part (usually the actual error message)
        lines = error.split('\n')
        if len(lines) > 20:
            error = '\n'.join(lines[-20:])
            parts.append("[WARNING] Traceback truncated (showing last 20 lines)")
        else:
            error = error[:max_display] + "... [TRUNCATED]"

    parts.append(error)

    # Add helpful hints
    hints = _generate_error_hints(error, exit_code)
    if hints:
        parts.append("")
        parts.append("[HINT] Hints:")
        for hint in hints:
            parts.append(f"  - {hint}")

    return "\n".join(parts)


def _generate_error_hints(error: str, exit_code: int) -> list[str]:
    """Generate helpful hints based on error type."""
    hints = []

    if exit_code == -1:
        hints.append("Process was killed (likely timeout or memory limit)")
        hints.append("Consider: reducing loop iterations, using efficient algorithms")

    if "MemoryError" in error:
        hints.append("Memory exhausted — avoid creating huge data structures")
        hints.append("Consider: processing data in chunks, using generators")

    if "TimeoutError" in error or exit_code == 124:
        hints.append("Execution timed out — code took too long to run")
        hints.append("Consider: optimizing algorithms, reducing data size")

    if "SyntaxError" in error:
        hints.append("Python syntax error — check your code structure")
        hints.append("Consider: using a code formatter, checking indentation")

    if "NameError" in error:
        hints.append("Undefined variable or function name")
        hints.append("Consider: checking variable names, importing required modules")

    if "ImportError" in error or "ModuleNotFoundError" in error:
        hints.append("Module not found or import failed")
        hints.append("Consider: using only standard library modules (os, sys, subprocess are banned)")

    if "RecursionError" in error:
        hints.append("Maximum recursion depth exceeded")
        hints.append("Consider: using iterative approach instead of recursion")

    if "ZeroDivisionError" in error:
        hints.append("Division by zero")
        hints.append("Consider: adding zero checks before division")

    if "KeyError" in error:
        hints.append("Dictionary key not found")
        hints.append("Consider: using dict.get() with default value")

    if "IndexError" in error:
        hints.append("List index out of range")
        hints.append("Consider: checking list length before indexing")

    return hints


# ─── Self-test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Output Sanitizer Self-Test ===\n")

    # Test 1: Normal text output
    result = sanitize_output("Hello, world!\n42\n{'status': 'ok'}")
    print(f"Test 1 - Text: {result.content_type} ({result.report.sanitized_length} chars)")
    assert result.content_type == "text"
    print("  PASS\n")

    # Test 1b: Pure JSON output
    result = sanitize_output('{"status": "ok", "value": 42}')
    print(f"Test 1b - JSON: {result.content_type}")
    assert result.content_type == "json"
    print("  PASS\n")

    # Test 2: Error output
    result = sanitize_output("Traceback (most recent call last):\n  File '<string>', line 1\n    x = 1/0\nZeroDivisionError: division by zero")
    print(f"Test 2 - Error: {result.content_type}")
    assert result.content_type == "error"
    print("  PASS\n")

    # Test 3: Truncation
    long_output = "A" * 2_000_000
    result = sanitize_output(long_output, max_length=1_000_000)
    print(f"Test 3 - Truncation: {result.report.was_truncated}, length={result.report.sanitized_length}")
    assert result.report.was_truncated
    assert result.report.sanitized_length <= 1_000_000 + len(TRUNCATION_MARKER)
    print("  PASS\n")

    # Test 4: Sensitive data redaction
    leaky_output = """
Processing data...
API_KEY = "sk-1234567890abcdef1234567890abcdef"
User email: john.doe@example.com
Credit card: 4111111111111111
Done.
"""
    result = sanitize_output(leaky_output)
    print(f"Test 4 - Redaction: {result.report.redaction_count} redactions, issues: {[i['type'] for i in result.report.issues_found]}")
    assert "sk-1234567890abcdef" not in result.content
    assert "john.doe@example.com" not in result.content
    assert "4111111111111111" not in result.content
    assert result.report.redaction_count >= 3
    print("  PASS\n")

    # Test 5: LLM formatting
    result = sanitize_output("Hello, world!")
    formatted = format_for_llm(result)
    print(f"Test 5 - LLM format:\n{formatted}")
    assert "[TEXT] Text Output:" in formatted
    print("  PASS\n")

    # Test 6: Error hints
    error_output = format_error_for_llm("ZeroDivisionError: division by zero", 1)
    print(f"Test 6 - Error hints:\n{error_output}")
    assert "division by zero" in error_output
    assert "[HINT] Hints:" in error_output
    print("  PASS\n")

    print("All tests passed!")
