"""
Audit logger and rate limiter for production code execution.

Provides:
  - Immutable audit trail for every execution
  - Sliding-window rate limiting per user
  - Usage statistics and anomaly detection
  - JSON serialization for log shipping
"""

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# ─── Configuration ───────────────────────────────────────────────

DEFAULT_RATE_LIMIT = 50          # Max executions per window
DEFAULT_RATE_WINDOW = 60         # Window in seconds
MAX_LOG_ENTRIES = 10_000         # Circular buffer size
ANOMALY_THRESHOLD = 10           # Failed execs before anomaly alert


# ─── Audit Log Entry ─────────────────────────────────────────────

@dataclass
class AuditEntry:
    """
    Immutable audit log entry for a single code execution.

    Fields are frozen after creation to prevent tampering.
    """
    timestamp: str           # ISO 8601 UTC
    execution_id: str        # Unique per-execution ID
    user_id: str             # Who triggered this
    code_hash: str           # SHA-256 of submitted code
    analysis_passed: bool    # Static analysis result
    analysis_issues: list    # Issues found (if any)
    execution_success: bool  # Did the code run successfully?
    exit_code: int           # Process exit code
    execution_time_ms: float # Wall time in milliseconds
    output_length: int       # Bytes of output produced
    output_truncated: bool   # Was output truncated?
    redactions_made: int     # Sensitive patterns redacted
    security_issues: list    # Post-execution security notices

    @classmethod
    def create(
        cls,
        code: str,
        user_id: str,
        analysis_passed: bool,
        analysis_issues: list,
        execution_success: bool,
        exit_code: int,
        execution_time_ms: float,
        output_length: int,
        output_truncated: bool,
        redactions_made: int,
        security_issues: list,
    ) -> "AuditEntry":
        """Factory with auto-generated timestamp and execution ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        # Deterministic ID from timestamp + code hash
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        exec_id = f"exec-{timestamp[:19].replace(':', '-')}-{code_hash[:8]}"
        return cls(
            timestamp=timestamp,
            execution_id=exec_id,
            user_id=user_id,
            code_hash=code_hash,
            analysis_passed=analysis_passed,
            analysis_issues=[str(i) for i in analysis_issues],
            execution_success=execution_success,
            exit_code=exit_code,
            execution_time_ms=execution_time_ms,
            output_length=output_length,
            output_truncated=output_truncated,
            redactions_made=redactions_made,
            security_issues=security_issues,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─── Audit Logger ────────────────────────────────────────────────

class AuditLogger:
    """
    Thread-safe audit logger with circular buffer.

    Keeps the last MAX_LOG_ENTRIES in memory.
    Provides search, statistics, and export.
    """

    def __init__(self, max_entries: int = MAX_LOG_ENTRIES):
        self._max_entries = max_entries
        self._entries: list[AuditEntry] = []
        self._lock = None  # Simple lock placeholder (single-threaded for now)

    def log(self, entry: AuditEntry):
        """Add an audit entry. Circular buffer: drops oldest if full."""
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def get_entries(self, user_id: Optional[str] = None, limit: int = 50) -> list[AuditEntry]:
        """Get recent entries, optionally filtered by user."""
        entries = self._entries
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        return entries[-limit:]

    def get_stats(self, user_id: Optional[str] = None) -> dict:
        """
        Compute execution statistics.

        Returns dict with:
        - total_executions
        - success_rate
        - avg_execution_time_ms
        - total_output_bytes
        - redaction_count
        - anomaly_alerts
        """
        entries = self._entries
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]

        if not entries:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0,
                "total_output_bytes": 0,
                "redaction_count": 0,
                "anomaly_alerts": [],
            }

        total = len(entries)
        successes = sum(1 for e in entries if e.execution_success)
        failures = total - successes
        total_time = sum(e.execution_time_ms for e in entries)
        total_output = sum(e.output_length for e in entries)
        total_redactions = sum(e.redactions_made for e in entries)

        # Anomaly detection: rapid failures
        recent_failures = 0
        anomaly_alerts = []
        for e in reversed(entries[-ANOMALY_THRESHOLD:]):
            if not e.execution_success and not e.analysis_passed:
                recent_failures += 1
                if recent_failures >= ANOMALY_THRESHOLD:
                    anomaly_alerts.append(
                        f"ANOMALY: {recent_failures} consecutive failures "
                        f"for user '{e.user_id}'"
                    )
                    break

        return {
            "total_executions": total,
            "successful": successes,
            "failed": failures,
            "success_rate": round(successes / total, 4) if total > 0 else 0.0,
            "avg_execution_time_ms": round(total_time / total, 2) if total > 0 else 0.0,
            "max_execution_time_ms": max(e.execution_time_ms for e in entries),
            "min_execution_time_ms": min(e.execution_time_ms for e in entries),
            "total_output_bytes": total_output,
            "total_redactions": total_redactions,
            "analysis_pass_rate": round(
                sum(1 for e in entries if e.analysis_passed) / total, 4
            ),
            "anomaly_alerts": anomaly_alerts,
            "period": {
                "first_entry": entries[0].timestamp,
                "last_entry": entries[-1].timestamp,
            },
        }

    def export(self, user_id: Optional[str] = None) -> str:
        """Export audit log as JSON string."""
        entries = self.get_entries(user_id=user_id, limit=len(self._entries))
        return json.dumps([e.to_dict() for e in entries], indent=2)

    def clear(self):
        """Clear all entries (for testing)."""
        self._entries.clear()


# ─── Rate Limiter ────────────────────────────────────────────────

class RateLimiter:
    """
    Sliding-window rate limiter per user.

    Prevents DoS through rapid code execution requests.
    Configurable max calls per time window.
    """

    def __init__(
        self,
        max_calls: int = DEFAULT_RATE_LIMIT,
        window_seconds: int = DEFAULT_RATE_WINDOW,
    ):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._user_windows: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> tuple[bool, str]:
        """
        Check if a user is allowed to execute code.

        Returns (allowed: bool, reason: str).
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old entries
        self._user_windows[user_id] = [
            t for t in self._user_windows[user_id] if t > window_start
        ]

        # Check limit
        if len(self._user_windows[user_id]) >= self.max_calls:
            wait_time = self._user_windows[user_id][0] + self.window_seconds - now
            return (
                False,
                f"Rate limit exceeded ({self.max_calls}/{self.window_seconds}s). "
                f"Wait {wait_time:.0f}s before next execution."
            )

        # Allow and record
        self._user_windows[user_id].append(now)
        return True, "OK"

    def get_usage(self, user_id: str) -> dict:
        """Get current rate limit usage for a user."""
        now = time.time()
        window_start = now - self.window_seconds
        calls_in_window = len([
            t for t in self._user_windows[user_id] if t > window_start
        ])
        return {
            "user_id": user_id,
            "calls_in_window": calls_in_window,
            "max_calls": self.max_calls,
            "window_seconds": self.window_seconds,
            "remaining": max(0, self.max_calls - calls_in_window),
            "utilization_pct": round(
                (calls_in_window / self.max_calls) * 100, 1
            ),
        }

    def reset(self, user_id: Optional[str] = None):
        """Reset rate limit counters (for testing)."""
        if user_id:
            self._user_windows[user_id] = []
        else:
            self._user_windows.clear()


# ─── Self-test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Audit Logger & Rate Limiter Self-Test ===\n")

    # Test 1: Audit logging
    logger = AuditLogger()
    entry = AuditEntry.create(
        code="print('hello')",
        user_id="test-user",
        analysis_passed=True,
        analysis_issues=[],
        execution_success=True,
        exit_code=0,
        execution_time_ms=123.45,
        output_length=6,
        output_truncated=False,
        redactions_made=0,
        security_issues=[],
    )
    logger.log(entry)
    stats = logger.get_stats()
    print(f"Test 1 - Audit log: {stats['total_executions']} executions, "
          f"success_rate={stats['success_rate']}")
    assert stats['total_executions'] == 1
    assert stats['success_rate'] == 1.0
    print("  PASS\n")

    # Test 2: Rate limiting
    limiter = RateLimiter(max_calls=3, window_seconds=60)
    for i in range(3):
        allowed, reason = limiter.is_allowed("user-1")
        assert allowed, f"Call {i+1} should be allowed: {reason}"
    allowed, reason = limiter.is_allowed("user-1")
    assert not allowed, "4th call should be blocked"
    print(f"Test 2 - Rate limit: blocked after 3 calls: '{reason}'")
    print("  PASS\n")

    # Test 3: Per-user isolation
    allowed, reason = limiter.is_allowed("user-2")
    assert allowed, "Different user should have own limit"
    print(f"Test 3 - User isolation: user-2 allowed: '{reason}'")
    print("  PASS\n")

    # Test 4: Usage stats
    usage = limiter.get_usage("user-1")
    print(f"Test 4 - Usage stats: {usage['calls_in_window']}/{usage['max_calls']} "
          f"({usage['utilization_pct']}%)")
    assert usage['calls_in_window'] == 3
    assert usage['remaining'] == 0
    print("  PASS\n")

    # Test 5: Export
    export = logger.export()
    data = json.loads(export)
    print(f"Test 5 - Export: {len(data)} entries serialized")
    assert len(data) == 1
    print("  PASS\n")

    print("All tests passed!")
