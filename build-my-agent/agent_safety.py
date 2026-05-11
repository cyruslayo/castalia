"""
Agent Safety & Guardrails — production-style safety layer for agents.

Implements Notebook 24 concepts in a reusable, testable module:
  - Input sanitization / prompt-injection detection
  - Tool validation / whitelisting / bounds / approvals
  - Output filtering / PII redaction
  - Request + tool rate limiting
  - Audit logging
  - Unified GuardrailsLayer wrapper
  - Maker-checker safety pattern
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from audit_logger import RateLimiter
from output_sanitizer import sanitize_output
from tool_registry import ToolRegistry, ToolResult, validate_with_helpful_errors
from tool_definitions import ToolDefinition


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SafetyIssue:
    category: str
    message: str
    matched: Optional[str] = None
    severity: str = "warning"

    def to_dict(self) -> dict:
        d = {
            "category": self.category,
            "message": self.message,
            "severity": self.severity,
        }
        if self.matched is not None:
            d["matched"] = self.matched
        return d


@dataclass
class ToolPolicy:
    """Extra runtime controls layered on top of ToolDefinition validation."""

    name: str
    enabled: bool = True
    requires_approval: bool = False
    max_calls_per_window: int = 60
    window_seconds: int = 60
    custom_validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None


@dataclass
class AuditEvent:
    timestamp: float
    event_type: str
    allowed: bool
    actor: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "allowed": self.allowed,
            "actor": self.actor,
            "details": self.details,
        }


# ============================================================================
# Input validation
# ============================================================================

class InputValidator:
    """Detect likely prompt injection, jailbreak, and stuffing attempts."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now\s+",
        r"new\s+instructions?\s*:",
        r"system\s*:",
        r"\[\s*system\s*\]",
        r"<\s*system\s*>",
        r"developer\s*:",
        r"assistant\s*:",
        r"admin\s+override",
        r"jailbreak",
        r"dan\s+mode",
        r"bypass\s+(all\s+)?restrictions",
        r"reveal\s+(your\s+)?system\s+prompt",
    ]

    SQL_INJECTION_PATTERNS = [
        r"(?:'|\"|\)|;|--)\s*(drop|delete|insert|update|alter|truncate)\b",
        r"union\s+select",
        r"'\s*(or|and)\s+'[^']+'\s*=\s*'[^']+'",
    ]

    def __init__(self, max_length: int = 10_000):
        self.max_length = max_length
        self._compiled_injection = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self._compiled_sql = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS]
        self.scan_log: List[dict] = []
        self.blocked_count = 0

    def validate(self, text: str) -> Dict[str, Any]:
        issues: List[SafetyIssue] = []

        for pattern in self._compiled_injection:
            match = pattern.search(text)
            if match:
                issues.append(SafetyIssue(
                    category="prompt_injection",
                    message="Prompt-injection pattern detected",
                    matched=match.group(0),
                    severity="high",
                ))

        for pattern in self._compiled_sql:
            match = pattern.search(text)
            if match:
                issues.append(SafetyIssue(
                    category="sql_injection",
                    message="Likely SQL-injection payload detected",
                    matched=match.group(0),
                    severity="high",
                ))

        if len(text) > self.max_length:
            issues.append(SafetyIssue(
                category="length_attack",
                message=f"Input length {len(text)} exceeds max {self.max_length}",
                severity="medium",
            ))

        safe = len(issues) == 0
        if not safe:
            self.blocked_count += 1

        result = {
            "safe": safe,
            "issues": [i.to_dict() for i in issues],
            "text_length": len(text),
        }
        self.scan_log.append(result)
        return result


# ============================================================================
# Tool validation
# ============================================================================

class ToolValidator:
    """Whitelist tools, validate parameters, enforce approvals and rate limits."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry
        self.policies: Dict[str, ToolPolicy] = {}
        self._rate_limiters: Dict[Tuple[str, str], RateLimiter] = {}
        self.call_log: List[dict] = []

    def register_policy(self, policy: ToolPolicy) -> None:
        self.policies[policy.name] = policy

    def _get_limiter(self, actor: str, policy: ToolPolicy) -> RateLimiter:
        key = (actor, policy.name)
        if key not in self._rate_limiters:
            self._rate_limiters[key] = RateLimiter(
                max_calls=policy.max_calls_per_window,
                window_seconds=policy.window_seconds,
            )
        return self._rate_limiters[key]

    def validate_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        actor: str = "system",
        approved: bool = False,
    ) -> Dict[str, Any]:
        issues: List[str] = []

        if tool_name not in self.policies:
            issues.append(f"Tool '{tool_name}' is not whitelisted")
            result = {"allowed": False, "issues": issues, "requires_approval": False}
            self.call_log.append({"tool": tool_name, "actor": actor, **result})
            return result

        policy = self.policies[tool_name]
        if not policy.enabled:
            issues.append(f"Tool '{tool_name}' is disabled by policy")

        if policy.requires_approval and not approved:
            issues.append(f"Tool '{tool_name}' requires human approval")

        if self.registry and tool_name in self.registry.definitions:
            definition: ToolDefinition = self.registry.definitions[tool_name]
            validation_error = validate_with_helpful_errors(definition, params)
            if validation_error:
                issues.append(validation_error)

        if policy.custom_validator is not None:
            extra_error = policy.custom_validator(params)
            if extra_error:
                issues.append(extra_error)

        # Only consume rate-limit budget if the request is otherwise valid.
        if not issues:
            limiter = self._get_limiter(actor, policy)
            allowed_rate, reason = limiter.is_allowed(actor)
            if not allowed_rate:
                issues.append(reason)

        allowed = len(issues) == 0
        result = {
            "allowed": allowed,
            "issues": issues,
            "requires_approval": policy.requires_approval,
        }
        self.call_log.append({"tool": tool_name, "actor": actor, "params": params, **result})
        return result


# ============================================================================
# Output filtering
# ============================================================================

class OutputFilter:
    """Redact PII / secrets from agent responses."""

    KEYWORD_PATTERNS = [
        re.compile(r"\b(password|secret|api[_ -]?key|private key|access token|bearer token)\b", re.IGNORECASE),
    ]

    def __init__(self):
        self.detections: List[dict] = []

    def scan(self, text: str) -> Dict[str, Any]:
        sanitized = sanitize_output(text)
        findings = list(sanitized.report.issues_found)

        for pattern in self.KEYWORD_PATTERNS:
            for match in pattern.finditer(text):
                findings.append({
                    "type": "SENSITIVE_KEYWORD",
                    "description": "Sensitive keyword present",
                    "position": match.start(),
                    "matched": match.group(0),
                })

        return {
            "clean": len(findings) == 0,
            "findings": findings,
            "count": len(findings),
            "sanitized": sanitized,
        }

    def filter(self, text: str) -> Tuple[str, Dict[str, Any]]:
        result = self.scan(text)
        sanitized = result["sanitized"]
        filtered = sanitized.content

        for pattern in self.KEYWORD_PATTERNS:
            filtered = pattern.sub("[REDACTED:SENSITIVE_KEYWORD]", filtered)

        result = {
            "clean": result["clean"],
            "findings": result["findings"],
            "count": result["count"],
            "redacted_text": filtered,
        }
        self.detections.append(result)
        return filtered, result


# ============================================================================
# Audit logging
# ============================================================================

class SafetyAuditLogger:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.events: List[AuditEvent] = []

    def log(self, event_type: str, allowed: bool, actor: str, **details) -> None:
        self.events.append(AuditEvent(
            timestamp=time.time(),
            event_type=event_type,
            allowed=allowed,
            actor=actor,
            details=details,
        ))

    def summary(self) -> Dict[str, Any]:
        by_type = defaultdict(int)
        blocked = 0
        for event in self.events:
            by_type[event.event_type] += 1
            if not event.allowed:
                blocked += 1
        return {
            "agent": self.agent_name,
            "total_events": len(self.events),
            "blocked_events": blocked,
            "by_type": dict(by_type),
        }


# ============================================================================
# Unified safety layer
# ============================================================================

class GuardrailsLayer:
    """Wrap an agent or tool registry with notebook-24 style defenses."""

    def __init__(
        self,
        agent_name: str = "GuardedAgent",
        registry: Optional[ToolRegistry] = None,
        responder: Optional[Callable[[str], str]] = None,
        max_requests_per_window: int = 30,
        request_window_seconds: int = 60,
    ):
        self.agent_name = agent_name
        self.registry = registry
        self.responder = responder or (lambda text: f"Echo: {text}")
        self.input_validator = InputValidator()
        self.tool_validator = ToolValidator(registry=registry)
        self.output_filter = OutputFilter()
        self.request_limiter = RateLimiter(
            max_calls=max_requests_per_window,
            window_seconds=request_window_seconds,
        )
        self.audit = SafetyAuditLogger(agent_name)

    def register_tool_policy(self, policy: ToolPolicy) -> None:
        self.tool_validator.register_policy(policy)

    def check_input(self, user_input: str, actor: str = "user") -> Tuple[bool, Dict[str, Any]]:
        result = self.input_validator.validate(user_input)
        self.audit.log("input_validation", result["safe"], actor, issues=result["issues"], preview=user_input[:80])
        return result["safe"], result

    def check_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        actor: str = "agent",
        approved: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        result = self.tool_validator.validate_call(tool_name, params, actor=actor, approved=approved)
        self.audit.log("tool_validation", result["allowed"], actor, tool=tool_name, issues=result["issues"])
        return result["allowed"], result

    def filter_output(self, output: str, actor: str = "agent") -> Tuple[str, Dict[str, Any]]:
        filtered, result = self.output_filter.filter(output)
        self.audit.log("output_filter", result["clean"], actor, findings=result["findings"], redacted=filtered != output)
        return filtered, result

    def process_request(self, user_input: str, actor: str = "user") -> Dict[str, Any]:
        input_safe, input_result = self.check_input(user_input, actor=actor)
        if not input_safe:
            return {
                "blocked": True,
                "stage": "input_validation",
                "reason": input_result["issues"],
                "response": "Request blocked: potentially unsafe input detected.",
            }

        allowed, reason = self.request_limiter.is_allowed(actor)
        self.audit.log("request_rate_limit", allowed, actor, reason=reason)
        if not allowed:
            return {
                "blocked": True,
                "stage": "rate_limit",
                "reason": [reason],
                "response": "Request blocked: rate limit exceeded.",
            }

        response = self.responder(user_input)
        filtered, scan = self.filter_output(response, actor=self.agent_name)
        return {
            "blocked": False,
            "response": filtered,
            "original_response": response if filtered != response else None,
            "pii_redacted": not scan["clean"],
            "checks_passed": ["input", "rate_limit", "output"],
        }

    def execute_tool(
        self,
        tool_name: str,
        actor: str = "agent",
        approved: bool = False,
        **params,
    ) -> ToolResult:
        if not self.registry:
            return ToolResult(False, error="No tool registry configured", tool_name=tool_name)

        allowed, validation = self.check_tool_call(tool_name, params, actor=actor, approved=approved)
        if not allowed:
            return ToolResult(False, error="; ".join(validation["issues"]), tool_name=tool_name)

        result = self.registry.call(tool_name, **params)
        self.audit.log("tool_execution", result.success, actor, tool=tool_name, result=result.to_dict())
        return result


# ============================================================================
# Maker-checker pattern
# ============================================================================

class MakerCheckerPipeline:
    """Generate with one function, validate with another."""

    def __init__(
        self,
        maker: Callable[[str], str],
        checker: Callable[[str, str], Tuple[bool, str]],
    ):
        self.maker = maker
        self.checker = checker
        self.maker_calls = 0
        self.checker_calls = 0
        self.rejections = 0

    def execute(self, task: str, max_retries: int = 2) -> Dict[str, Any]:
        last_reason = ""
        for attempt in range(max_retries + 1):
            self.maker_calls += 1
            response = self.maker(task)
            self.checker_calls += 1
            approved, reason = self.checker(task, response)
            if approved:
                return {
                    "approved": True,
                    "response": response,
                    "attempts": attempt + 1,
                    "review": reason,
                }
            self.rejections += 1
            last_reason = reason

        return {
            "approved": False,
            "response": "Unable to generate an approved response.",
            "attempts": max_retries + 1,
            "review": last_reason,
        }
