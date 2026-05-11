from runtime_contracts import AgentRequest, AgentResult
from evaluation_runtime import (
    AgentEvaluator,
    CostTracker,
    GoldenTask,
    RegressionRunner,
    RegressionTracker,
    build_default_golden_dataset,
    default_judge,
    normalize_answer,
    score_exact_match,
    score_fuzzy_match,
    score_tool_accuracy,
    score_with_judge,
)


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.calls.append(request)
        if request.task == "math":
            return AgentResult(
                answer="The answer is 7981.",
                success=True,
                strategy_used="react",
                steps=[{"step": 1}, {"step": 2}],
                tool_calls=[{"tool": "calculator", "params": {"expression": "347*23"}, "success": True}],
                metadata={"input_tokens": 10, "output_tokens": 5, "duration_seconds": 0.2},
            ).finish()
        if request.task == "fact":
            return AgentResult(
                answer="London",
                success=True,
                strategy_used="react",
                steps=[{"step": 1}],
                tool_calls=[{"tool": "search_kb", "params": {"query": "capital of france"}, "success": True}],
                metadata={"input_tokens": 12, "output_tokens": 4, "duration_seconds": 0.1},
            ).finish()
        if request.task == "blocked":
            return AgentResult(
                answer="Request blocked",
                success=False,
                strategy_used="blocked",
                errors=["unsafe input"],
                metadata={"duration_seconds": 0.05},
            ).finish()
        raise ValueError("unexpected task")


def test_scoring_helpers_cover_normalization_keywords_and_tools():
    assert normalize_answer("The answer is $12.68.") == "12.68"
    assert score_exact_match("42", "The answer is 42") == 1.0
    assert score_exact_match("42", "43") == 0.0
    assert score_fuzzy_match("George Orwell", "Orwell wrote it") == 0.5
    assert round(score_tool_accuracy(["calculator"], ["calculator", "web_search"]), 2) == 0.67


def test_default_judge_accepts_reference_embedded_in_longer_answer():
    score, reason = default_judge(
        "What is the capital of France?",
        "Paris",
        "Paris is the capital of France.",
    )
    assert score == 1.0
    assert "verbatim" in reason.lower()


def test_score_with_judge_accepts_structured_string_response():
    score, reason = score_with_judge(
        "What is 2+2?",
        "4",
        "4",
        judge_fn=lambda q, e, a: "Score: 5\nReason: Perfect answer.",
    )
    assert score == 1.0
    assert "Perfect" in reason


def test_default_golden_dataset_has_notebook_sized_coverage():
    dataset = build_default_golden_dataset()
    assert len(dataset) >= 15
    assert any(task.category == "math" for task in dataset)
    assert any(task.category == "factual" for task in dataset)
    assert any(task.expected_tools for task in dataset)


def test_agent_evaluator_scores_and_aggregates_fake_runtime():
    runtime = FakeRuntime()
    evaluator = AgentEvaluator(runtime, name="FakeRuntime", judge_fn=lambda q, e, a: (1.0 if "7981" in a else 0.0, "stub"))
    tasks = [
        GoldenTask("math-001", "math", expected_answer="7981", expected_tools=["calculator"], category="math"),
        GoldenTask("fact-001", "fact", expected_answer="Paris", expected_tools=["search_kb"], category="factual"),
    ]

    results = evaluator.evaluate_dataset(tasks)
    report = evaluator.aggregate_report()

    assert len(results) == 2
    assert results[0].passed is True
    assert results[0].tool_accuracy == 1.0
    assert results[1].passed is False
    assert report["total_tasks"] == 2
    assert report["passed"] == 1
    assert report["pass_rate"] == 0.5
    assert "math" in report["by_category"]
    assert "factual" in report["by_category"]


def test_cost_tracker_summarizes_eval_metrics():
    runtime = FakeRuntime()
    evaluator = AgentEvaluator(runtime, judge_fn=lambda q, e, a: (1.0, "ok"))
    metrics = evaluator.evaluate_single(GoldenTask("math-001", "math", expected_answer="7981", expected_tools=["calculator"]))

    tracker = CostTracker()
    tracker.record(metrics)
    summary = tracker.summary()

    assert summary["num_tasks"] == 1
    assert summary["total_tokens"] == 15
    assert summary["avg_latency"] == 0.2
    assert summary["total_estimated_cost"] > 0


def test_regression_tracker_detects_quality_regression_and_cost_improvement(tmp_path):
    tracker = RegressionTracker()
    baseline = {
        "pass_rate": 1.0,
        "avg_exact_match": 1.0,
        "avg_fuzzy_score": 1.0,
        "avg_judge_score": 1.0,
        "avg_tool_accuracy": 1.0,
        "avg_composite": 1.0,
        "avg_tokens_per_task": 100.0,
        "avg_latency": 2.0,
    }
    tracker.save_baseline("v1", baseline)

    path = tmp_path / "baseline.json"
    tracker.save_baseline_file(str(path), baseline)
    loaded = tracker.load_baseline_file(str(path), name="v1-file")
    assert loaded["report"]["pass_rate"] == 1.0

    current = {**baseline, "avg_exact_match": 0.8, "avg_composite": 0.85, "avg_tokens_per_task": 80.0}
    comparison = tracker.compare("v1", current, threshold=0.05)

    assert "avg_exact_match" in comparison["regressions"]
    assert comparison["metrics"]["avg_tokens_per_task"]["status"] == "improved"


def test_regression_runner_returns_backwards_compatible_suite_shape():
    runner = RegressionRunner(FakeRuntime(), judge_fn=lambda q, e, a: (1.0 if "7981" in a else 0.0, "stub"))
    suite = runner.run_suite([
        GoldenTask("math-001", "math", expected_answer="7981", expected_tools=["calculator"]),
    ])

    assert suite["passed"] == 1
    assert suite["total"] == 1
    assert suite["results"][0]["task_id"] == "math-001"
    assert "report" in suite
