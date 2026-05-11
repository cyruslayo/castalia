from agent_runtime import AgentRuntime
from lifecycle_runtime import AgentLogger, AgentRegistry
from runtime_contracts import AgentContext, AgentRequest, AgentResult


class EchoAdapter:
    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        return AgentResult(
            answer=f"done: {request.task}",
            success=True,
            strategy_used="react",
            metadata={"echo": True},
            request_id=request.request_id,
        ).finish()


class FailingAdapter:
    def run(self, request: AgentRequest, context: AgentContext) -> AgentResult:
        raise RuntimeError("boom")


def test_agent_logger_keeps_traces_and_exports_json():
    logger = AgentLogger()
    logger.log("researcher", "INFO", "started")
    logger.log("researcher", "ERROR", "failed", {"reason": "boom"})

    trace = logger.trace("researcher")
    assert len(trace) == 2
    assert logger.errors()[0]["message"] == "failed"
    exported = logger.export_json()
    assert '"researcher"' in exported
    assert '"failed"' in exported


def test_agent_registry_and_lifecycle_round_trip():
    runtime = AgentRuntime(adapters={"react": EchoAdapter(), "auto": EchoAdapter()}, runtime_config={"max_steps": 1, "verbose": False})
    runtime.register_agent("researcher", strategy="react", capabilities=["search", "summarize"])

    status = runtime.get_status()
    assert status["registered_agents"] == 1
    assert status["running_agents"] == 1
    assert runtime.health_check("researcher")["healthy"] is True

    result = runtime.submit_task("researcher", "hello")
    assert result.success is True
    assert result.answer == "done: hello"
    assert runtime.health_check("researcher")["tasks_completed"] == 1

    runtime.restart_agent("researcher")
    assert runtime.health_check("researcher")["restart_count"] == 1

    runtime.stop_agent("researcher")
    stopped = runtime.health_check("researcher")
    assert stopped["status"] == "stopped"
    blocked = runtime.submit_task("researcher", "again")
    assert blocked.success is False
    assert "not running" in blocked.answer


def test_agent_runtime_marks_lifecycle_failure_and_logs_error():
    runtime = AgentRuntime(adapters={"react": FailingAdapter(), "auto": FailingAdapter()}, runtime_config={"max_steps": 1, "verbose": False})
    runtime.register_agent("researcher", strategy="react")

    result = runtime.submit_task("researcher", "explode")
    assert result.success is False
    health = runtime.health_check("researcher")
    assert health["status"] == "failed"
    assert health["errors"] >= 1
    assert "RuntimeError: boom" in (health["last_error"] or "")
    assert runtime.get_status()["log_stats"]["by_level"]["ERROR"] >= 1
