"""Resilience primitives for Notebook 28 foundation.

Implements the five patterns from `agents/28_error_handling_and_resilience.ipynb`:
- RetryWithFeedback
- FallbackChain
- CircuitBreaker
- GracefulDegradation
- timeout handling

Also provides SafeToolExecutor, which combines timeout + retry + circuit breaker.
"""

from __future__ import annotations

import ctypes
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple


class TimeoutError(Exception):
    """Raised when a function call exceeds its time limit."""


@dataclass
class RetryRecord:
    attempt: int
    status: str
    error: Optional[str] = None


class RetryWithFeedback:
    """Retry failed operations and optionally feed failure context back in.

    Two modes are supported:
    - `execute(fn, messages)` for notebook-style callables that accept message history
      and return `(result, success)`.
    - `run(fn, *args, **kwargs)` for generic callables that should be retried on exception.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        error_prefix: str = "Your previous attempt failed",
    ):
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.error_prefix = error_prefix
        self.retry_log: List[Dict[str, Any]] = []

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1

    def execute(
        self,
        fn: Callable[[List[Dict[str, Any]]], Tuple[Any, bool]],
        messages: List[Dict[str, Any]],
        error_prefix: Optional[str] = None,
    ) -> Tuple[Any, int]:
        """Execute a notebook-style callable with retry feedback.

        Returns `(result, retries_used)`.
        """
        prefix = error_prefix or self.error_prefix
        current_messages = list(messages)
        last_error: Any = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result, success = fn(current_messages)
                if success:
                    self.retry_log.append({"attempt": attempt, "status": "success"})
                    return result, attempt - 1

                last_error = result
                self.retry_log.append(
                    {"attempt": attempt, "status": "retry", "error": str(last_error)[:200]}
                )
                if attempt < self.max_attempts:
                    feedback = (
                        f"{prefix} with error: {last_error}\n"
                        f"Please try again with a corrected approach."
                    )
                    current_messages = current_messages + [
                        {"role": "assistant", "content": str(last_error)},
                        {"role": "user", "content": feedback},
                    ]
                    self._sleep(attempt)
            except Exception as exc:
                last_error = str(exc)
                self.retry_log.append(
                    {"attempt": attempt, "status": "retry", "error": last_error[:200]}
                )
                if attempt < self.max_attempts:
                    feedback = (
                        f"{prefix} with exception: {last_error}\n"
                        f"Please try again, avoiding the error."
                    )
                    current_messages = current_messages + [{"role": "user", "content": feedback}]
                    self._sleep(attempt)

        return last_error, self.max_retries

    def run(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Retry a generic callable on exception and return its result."""
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = fn(*args, **kwargs)
                self.retry_log.append({"attempt": attempt, "status": "success"})
                return result
            except Exception as exc:
                last_error = exc
                self.retry_log.append(
                    {"attempt": attempt, "status": "retry", "error": str(exc)[:200]}
                )
                if attempt < self.max_attempts:
                    self._sleep(attempt)
        raise last_error

    def _sleep(self, attempt: int) -> None:
        delay = self.backoff_seconds * attempt
        if delay > 0:
            time.sleep(delay)

    def stats(self) -> str:
        successes = sum(1 for r in self.retry_log if r["status"] == "success")
        retries = sum(1 for r in self.retry_log if r["status"] == "retry")
        return (
            f"RetryWithFeedback: {successes} successes, {retries} retries "
            f"across {len(self.retry_log)} attempts"
        )


class FallbackChain:
    """Try multiple strategies in order until one succeeds."""

    def __init__(self, strategies: List[Any]):
        self.strategies = [self._normalize_strategy(s) for s in strategies]
        self.execution_log: List[Dict[str, Any]] = []

    def _normalize_strategy(self, strategy: Any) -> Dict[str, Any]:
        if isinstance(strategy, dict):
            if "fn" not in strategy:
                raise ValueError("Strategy dict must include 'fn'")
            return {
                "name": strategy.get("name") or getattr(strategy["fn"], "__name__", "strategy"),
                "fn": strategy["fn"],
            }
        if callable(strategy):
            return {"name": getattr(strategy, "__name__", "strategy"), "fn": strategy}
        raise TypeError("Each strategy must be a callable or {'name', 'fn'} dict")

    def execute(self, *args, **kwargs) -> Tuple[Any, str]:
        for strategy in self.strategies:
            name = strategy["name"]
            fn = strategy["fn"]
            try:
                result = fn(*args, **kwargs)
                success = True
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], bool):
                    result, success = result
                self.execution_log.append(
                    {"strategy": name, "status": "success" if success else "failed"}
                )
                if success:
                    return result, name
            except Exception as exc:
                self.execution_log.append(
                    {"strategy": name, "status": "error", "error": str(exc)[:200]}
                )
        return None, "all_failed"

    def run(self, *args, **kwargs) -> Any:
        result, strategy = self.execute(*args, **kwargs)
        if strategy == "all_failed":
            errors = [
                f"{entry['strategy']}: {entry.get('error', entry['status'])}"
                for entry in self.execution_log[-len(self.strategies):]
            ]
            raise RuntimeError("All fallbacks failed: " + "; ".join(errors))
        return result

    def stats(self) -> str:
        by_strategy: Dict[str, Dict[str, int]] = {}
        for entry in self.execution_log:
            counts = by_strategy.setdefault(entry["strategy"], {"success": 0, "failed": 0, "error": 0})
            counts[entry["status"]] += 1
        parts = [f"{name}: {counts}" for name, counts in by_strategy.items()]
        return "FallbackChain: " + " | ".join(parts)


class CircuitBreaker:
    """Circuit breaker with CLOSED → OPEN → HALF_OPEN transitions."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        cooldown_seconds: Optional[float] = None,
        recovery_seconds: Optional[float] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = recovery_seconds if recovery_seconds is not None else (cooldown_seconds or 30.0)
        self.state = self.CLOSED
        self.failure_count = 0
        self.failures = 0  # backward-compatible alias
        self.last_failure_time = 0.0
        self.opened_at: Optional[float] = None
        self.total_calls = 0
        self.total_blocked = 0
        self.state_history: List[Dict[str, Any]] = []

    def _transition(self, new_state: str, reason: str) -> None:
        self.state_history.append(
            {
                "from": self.state,
                "to": new_state,
                "reason": reason,
                "time": time.time(),
            }
        )
        self.state = new_state
        if new_state == self.OPEN:
            self.opened_at = time.time()

    def can_execute(self) -> bool:
        self.total_calls += 1
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.cooldown_seconds:
                self._transition(self.HALF_OPEN, f"Cooldown elapsed ({elapsed:.2f}s)")
                return True
            self.total_blocked += 1
            return False
        return True

    def record_success(self) -> None:
        if self.state == self.HALF_OPEN:
            self._transition(self.CLOSED, "Probe succeeded")
        self.failure_count = 0
        self.failures = 0

    def record_failure(self) -> None:
        self.failure_count += 1
        self.failures = self.failure_count
        self.last_failure_time = time.time()
        if self.state == self.HALF_OPEN:
            self._transition(self.OPEN, "Probe failed")
        elif self.failure_count >= self.failure_threshold and self.state != self.OPEN:
            self._transition(self.OPEN, f"{self.failure_count} consecutive failures")

    def execute(self, fn: Callable[..., Any], *args, **kwargs) -> Tuple[Any, bool]:
        if not self.can_execute():
            return None, False
        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result, True
        except Exception as exc:
            self.record_failure()
            return str(exc), False

    def call(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        result, success = self.execute(fn, *args, **kwargs)
        if success:
            return result
        if self.state == self.OPEN and result is None:
            raise RuntimeError("Circuit breaker is open")
        raise RuntimeError(str(result))

    def stats(self) -> str:
        return (
            f"CircuitBreaker '{self.name}': state={self.state} | "
            f"failures={self.failure_count}/{self.failure_threshold} | "
            f"calls={self.total_calls} | blocked={self.total_blocked} | "
            f"transitions={len(self.state_history)}"
        )


@dataclass
class PartialResult:
    step: str
    result: Any = None
    error: Optional[str] = None
    status: str = "success"


class GracefulDegradation:
    """Capture partial success and return the best available result."""

    def __init__(self):
        self.partial_results: List[Dict[str, Any]] = []
        self.failed_steps: List[Dict[str, Any]] = []

    def add_result(self, step_name: str, result: Any) -> None:
        self.partial_results.append({"step": step_name, "result": result, "status": "success"})

    def add_failure(self, step_name: str, error: str) -> None:
        self.failed_steps.append({"step": step_name, "error": error, "status": "failed"})

    @property
    def completion_rate(self) -> float:
        total = len(self.partial_results) + len(self.failed_steps)
        return len(self.partial_results) / total if total else 0.0

    def get_result(self) -> Dict[str, Any]:
        if not self.failed_steps:
            return {
                "status": "complete",
                "results": [r["result"] for r in self.partial_results],
                "completion_rate": 1.0 if self.partial_results else 0.0,
            }
        if self.partial_results:
            total = len(self.partial_results) + len(self.failed_steps)
            return {
                "status": "partial",
                "results": [r["result"] for r in self.partial_results],
                "failed_steps": [f["step"] for f in self.failed_steps],
                "completion_rate": self.completion_rate,
                "message": f"Completed {len(self.partial_results)}/{total} steps.",
            }
        return {
            "status": "failed",
            "results": [],
            "failed_steps": [f["step"] for f in self.failed_steps],
            "completion_rate": 0.0,
            "message": "All steps failed.",
        }


def _async_raise(thread_id: int, exc_type: type[BaseException]) -> None:
    """Raise an exception in another thread.

    Used only for timeout cancellation of worker threads. This is not suitable for
    arbitrary production cancellation, but is sufficient for deterministic teaching
    examples and tests where the worker is sleeping or CPU-bound Python code.
    """
    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id), ctypes.py_object(exc_type)
    )
    if result > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
        raise RuntimeError("Failed to cancel timed-out thread")


@dataclass
class _CallState:
    result: Any = None
    error: Optional[BaseException] = None
    done: threading.Event = field(default_factory=threading.Event)
    thread_id: Optional[int] = None


def call_with_timeout(fn: Callable[..., Any], seconds: float, *args, **kwargs) -> Any:
    """Run a callable with a timeout on all platforms."""
    if seconds is None or seconds <= 0:
        return fn(*args, **kwargs)

    state = _CallState()

    def target() -> None:
        state.thread_id = threading.get_ident()
        try:
            state.result = fn(*args, **kwargs)
        except BaseException as exc:  # pragma: no cover - exercised indirectly
            state.error = exc
        finally:
            state.done.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    finished = state.done.wait(seconds)
    if not finished:
        if state.thread_id is not None:
            try:
                _async_raise(state.thread_id, TimeoutError)
            except Exception:
                pass
        thread.join(min(0.1, seconds))
        raise TimeoutError(f"Function '{getattr(fn, '__name__', 'callable')}' timed out after {seconds}s")
    if state.error is not None:
        raise state.error
    return state.result


def with_timeout(seconds: float) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator adding timeout protection to any callable."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return call_with_timeout(fn, seconds, *args, **kwargs)

        return wrapper

    return decorator


class SafeToolExecutor:
    """Execute tools with timeout, retry, and circuit breaker protection."""

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        circuit_threshold: int = 3,
        circuit_cooldown_seconds: float = 30.0,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.circuit_threshold = circuit_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.execution_log: List[Dict[str, Any]] = []

    def _get_breaker(self, tool_name: str) -> CircuitBreaker:
        if tool_name not in self.breakers:
            self.breakers[tool_name] = CircuitBreaker(
                name=tool_name,
                failure_threshold=self.circuit_threshold,
                cooldown_seconds=self.circuit_cooldown_seconds,
            )
        return self.breakers[tool_name]

    def execute(self, tool_name: str, tool_fn: Callable[..., Any], *args, **kwargs) -> Tuple[Any, bool]:
        breaker = self._get_breaker(tool_name)
        if not breaker.can_execute():
            self.execution_log.append({"tool": tool_name, "status": "circuit_open"})
            return f"Circuit open for {tool_name}", False

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                result = call_with_timeout(tool_fn, self.timeout_seconds, *args, **kwargs)
                breaker.record_success()
                self.execution_log.append(
                    {"tool": tool_name, "status": "success", "attempts": attempt}
                )
                return result, True
            except Exception as exc:
                last_error = exc
                if attempt > self.max_retries:
                    breaker.record_failure()
                    self.execution_log.append(
                        {
                            "tool": tool_name,
                            "status": "failed",
                            "error": str(exc)[:200],
                            "attempts": attempt,
                        }
                    )
                    return str(exc), False

        return str(last_error or "Max retries exceeded"), False

    def stats(self) -> str:
        success = sum(1 for e in self.execution_log if e["status"] == "success")
        failed = sum(1 for e in self.execution_log if e["status"] == "failed")
        blocked = sum(1 for e in self.execution_log if e["status"] == "circuit_open")
        return f"SafeToolExecutor: {success} success, {failed} failed, {blocked} blocked"
