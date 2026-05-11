import time

from resilience import (
    CircuitBreaker,
    FallbackChain,
    GracefulDegradation,
    RetryWithFeedback,
    SafeToolExecutor,
    TimeoutError,
    call_with_timeout,
    with_timeout,
)


def test_retry_with_feedback_execute_self_corrects_after_error():
    retry = RetryWithFeedback(max_retries=2, backoff_seconds=0)
    seen = {"calls": 0}

    def fn(messages):
        seen["calls"] += 1
        if seen["calls"] == 1:
            return "parse failed", False
        assert "parse failed" in messages[-2]["content"]
        return {"answer": "ok"}, True

    result, retries_used = retry.execute(fn, [{"role": "user", "content": "hi"}])
    assert result == {"answer": "ok"}
    assert retries_used == 1
    assert seen["calls"] == 2


def test_retry_with_feedback_run_retries_exceptions():
    retry = RetryWithFeedback(max_retries=2, backoff_seconds=0)
    seen = {"calls": 0}

    def fn():
        seen["calls"] += 1
        if seen["calls"] < 3:
            raise ValueError("boom")
        return 42

    assert retry.run(fn) == 42
    assert seen["calls"] == 3


def test_fallback_chain_uses_second_strategy_after_first_failure():
    chain = FallbackChain([
        {"name": "first", "fn": lambda: ("bad", False)},
        {"name": "second", "fn": lambda: ("good", True)},
    ])

    result, strategy = chain.execute()
    assert result == "good"
    assert strategy == "second"


def test_circuit_breaker_opens_then_recovers_to_half_open_then_closed():
    breaker = CircuitBreaker("test", failure_threshold=2, cooldown_seconds=0.01)

    assert breaker.execute(lambda: (_ for _ in ()).throw(ValueError("x")))[1] is False
    assert breaker.state == CircuitBreaker.CLOSED

    assert breaker.execute(lambda: (_ for _ in ()).throw(ValueError("y")))[1] is False
    assert breaker.state == CircuitBreaker.OPEN

    result, success = breaker.execute(lambda: "blocked")
    assert result is None
    assert success is False

    time.sleep(0.02)
    result, success = breaker.execute(lambda: "ok")
    assert result == "ok"
    assert success is True
    assert breaker.state == CircuitBreaker.CLOSED


def test_graceful_degradation_reports_partial_result():
    degradation = GracefulDegradation()
    degradation.add_result("step1", "a")
    degradation.add_failure("step2", "boom")

    result = degradation.get_result()
    assert result["status"] == "partial"
    assert result["results"] == ["a"]
    assert result["failed_steps"] == ["step2"]
    assert result["completion_rate"] == 0.5


def test_call_with_timeout_raises_on_slow_function():
    def slow():
        time.sleep(0.2)
        return "done"

    start = time.time()
    try:
        call_with_timeout(slow, 0.05)
        assert False, "expected timeout"
    except TimeoutError as exc:
        assert "timed out" in str(exc)
    assert time.time() - start < 0.2


def test_with_timeout_decorator_allows_fast_function():
    @with_timeout(0.2)
    def fast():
        return "ok"

    assert fast() == "ok"


def test_safe_tool_executor_retries_and_succeeds():
    executor = SafeToolExecutor(timeout_seconds=0.2, max_retries=1, circuit_threshold=2)
    seen = {"calls": 0}

    def flaky(x):
        seen["calls"] += 1
        if seen["calls"] == 1:
            raise ValueError("transient")
        return x * 2

    result, ok = executor.execute("flaky", flaky, 21)
    assert ok is True
    assert result == 42
    assert seen["calls"] == 2


def test_safe_tool_executor_opens_circuit_after_repeated_failures():
    executor = SafeToolExecutor(
        timeout_seconds=0.2,
        max_retries=0,
        circuit_threshold=2,
        circuit_cooldown_seconds=10,
    )

    def broken(_):
        raise RuntimeError("down")

    assert executor.execute("broken", broken, 1)[1] is False
    assert executor.execute("broken", broken, 1)[1] is False
    result, ok = executor.execute("broken", broken, 1)
    assert ok is False
    assert "Circuit open" in result
