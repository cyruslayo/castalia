"""Evaluation and regression runtime for integrated agents (Notebook 26).

This module lifts the ideas from the evaluation notebook into the shared runtime
layer so later notebooks can measure quality, cost, and regressions against one
canonical agent interface.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from runtime_contracts import AgentRequest


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for",
    "and", "or", "it", "that", "this", "be", "with", "on", "at", "by", "as",
    "from", "then", "than", "into", "about", "after", "before", "than",
}


@dataclass
class GoldenTask:
    """One curated task in an evaluation suite."""

    id: str
    task: str
    expected_answer: str = ""
    expected_contains: List[str] = field(default_factory=list)
    forbidden_contains: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    tools_expected: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"
    strategy: str = "auto"
    max_steps: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        merged = []
        for name in [*self.expected_tools, *self.tools_expected]:
            if name not in merged:
                merged.append(name)
        self.expected_tools = merged
        self.tools_expected = list(merged)


@dataclass
class EvalMetrics:
    """Multi-dimensional evaluation result for one task."""

    task_id: str
    task_description: str
    expected_answer: str
    actual_answer: str
    category: str = "general"
    difficulty: str = "medium"
    strategy_used: str = "unknown"
    request_id: Optional[str] = None
    exact_match: float = 0.0
    fuzzy_score: float = 0.0
    judge_score: float = 0.0
    contains_score: float = 1.0
    forbidden_score: float = 1.0
    tools_used: List[str] = field(default_factory=list)
    tools_expected: List[str] = field(default_factory=list)
    tool_accuracy: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    num_steps: int = 0
    completed: bool = False
    success: bool = False
    error: Optional[str] = None
    reasons: List[str] = field(default_factory=list)

    @property
    def content_score(self) -> float:
        active = []
        if self.contains_score is not None:
            active.append(self.contains_score)
        if self.forbidden_score is not None:
            active.append(self.forbidden_score)
        return sum(active) / len(active) if active else 1.0

    @property
    def composite_score(self) -> float:
        weighted_total = 0.0
        total_weight = 0.0

        if self.expected_answer:
            for weight, score in ((0.3, self.exact_match), (0.2, self.fuzzy_score), (0.3, self.judge_score)):
                weighted_total += weight * score
                total_weight += weight
        else:
            weighted_total += 0.8 * self.content_score
            total_weight += 0.8

        if self.tools_expected or self.tools_used:
            weighted_total += 0.2 * self.tool_accuracy
            total_weight += 0.2

        if total_weight == 0:
            return 1.0 if self.completed and self.success else 0.0
        return weighted_total / total_weight

    @property
    def passed(self) -> bool:
        if not (self.completed and self.success):
            return False
        answer_ok = True
        if self.expected_answer:
            answer_ok = self.exact_match == 1.0 or self.judge_score >= 0.95
        content_ok = self.contains_score == 1.0 and self.forbidden_score == 1.0
        tool_ok = self.tool_accuracy == 1.0 if (self.tools_expected or self.tools_used) else True
        return answer_ok and content_ok and tool_ok

    def summary(self) -> str:
        return (
            f"Task: {self.task_id} | Passed: {self.passed} | Composite: {self.composite_score:.2f} | "
            f"Exact: {self.exact_match:.0%} | Fuzzy: {self.fuzzy_score:.0%} | "
            f"Judge: {self.judge_score:.0%} | Tools: {self.tool_accuracy:.0%} | "
            f"Tokens: {self.total_tokens} | Steps: {self.num_steps}"
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["content_score"] = self.content_score
        data["composite_score"] = self.composite_score
        data["passed"] = self.passed
        return data


@dataclass
class EvalResult:
    """Backwards-compatible pass/fail summary for one task."""

    task_id: str
    passed: bool
    score: float
    reasons: List[str]
    answer: str
    strategy_used: str
    tools_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


JudgeFn = Callable[[str, str, str], Any]


def normalize_answer(text: Optional[str]) -> str:
    """Normalize an answer for exact comparison."""
    text = (text or "").lower().strip()
    for prefix in ["the answer is", "answer:", "result:", "final answer:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.replace("$", "").replace(",", "").rstrip(".")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_keywords(text: Optional[str]) -> set[str]:
    words = re.findall(r"\b\w+\b", (text or "").lower())
    return {word for word in words if word not in STOP_WORDS}


def _extract_numbers(text: Optional[str]) -> List[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", normalize_answer(text))


def score_exact_match(expected: str, actual: str) -> float:
    return 1.0 if normalize_answer(expected) == normalize_answer(actual) else 0.0


def score_fuzzy_match(expected: str, actual: str) -> float:
    expected_kw = _extract_keywords(expected)
    actual_kw = _extract_keywords(actual)
    if not expected_kw:
        return 1.0 if not actual_kw else 0.0
    return len(expected_kw & actual_kw) / len(expected_kw)


def score_tool_accuracy(expected_tools: List[str], actual_tools: List[str]) -> float:
    if not expected_tools and not actual_tools:
        return 1.0
    if not expected_tools and actual_tools:
        return 0.5
    if expected_tools and not actual_tools:
        return 0.0

    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    true_positive = len(expected_set & actual_set)
    precision = true_positive / len(actual_set) if actual_set else 0.0
    recall = true_positive / len(expected_set) if expected_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_contains_all(expected_contains: List[str], actual: str) -> Tuple[float, List[str]]:
    if not expected_contains:
        return 1.0, []
    haystack = (actual or "").lower()
    failures = []
    hits = 0
    for needle in expected_contains:
        ok = needle.lower() in haystack
        if ok:
            hits += 1
        else:
            failures.append(f"Missing expected text: {needle}")
    return hits / len(expected_contains), failures


def score_forbidden_absent(forbidden_contains: List[str], actual: str) -> Tuple[float, List[str]]:
    if not forbidden_contains:
        return 1.0, []
    haystack = (actual or "").lower()
    failures = []
    clean = 0
    for needle in forbidden_contains:
        ok = needle.lower() not in haystack
        if ok:
            clean += 1
        else:
            failures.append(f"Contained forbidden text: {needle}")
    return clean / len(forbidden_contains), failures


def default_judge(question: str, expected: str, actual: str) -> Tuple[float, str]:
    """Cheap deterministic judge used when no external LLM judge is provided."""
    exact = score_exact_match(expected, actual)
    fuzzy = score_fuzzy_match(expected, actual)
    numeric_match = 0.0
    expected_numbers = _extract_numbers(expected)
    actual_numbers = _extract_numbers(actual)
    normalized_expected = normalize_answer(expected)
    normalized_actual = normalize_answer(actual)
    if expected_numbers and actual_numbers and expected_numbers[0] == actual_numbers[0]:
        numeric_match = 1.0

    if exact == 1.0:
        return 1.0, "Exact normalized match."
    if normalized_expected and normalized_expected in normalized_actual:
        return 1.0, "Reference answer appears verbatim inside the assistant answer."

    score = max(exact, 0.6 * exact + 0.25 * fuzzy + 0.15 * numeric_match)
    if numeric_match == 1.0:
        reason = "Numeric answer matches after extraction."
    elif fuzzy > 0:
        reason = "Partial keyword overlap with the reference answer."
    else:
        reason = "Answer does not align with the reference answer."
    return max(0.0, min(1.0, score)), reason


def _call_judge(judge_fn: JudgeFn, question: str, expected: str, actual: str) -> Any:
    try:
        signature = inspect.signature(judge_fn)
        arity = len([p for p in signature.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        arity = 3

    if arity <= 1:
        prompt = (
            f"Question: {question}\n"
            f"Reference answer: {expected}\n"
            f"Assistant answer: {actual}\n"
            "Respond with 'Score: [1-5]' and 'Reason: ...'."
        )
        return judge_fn(prompt)
    return judge_fn(question, expected, actual)


def score_with_judge(question: str, expected: str, actual: str,
                     judge_fn: Optional[JudgeFn] = None) -> Tuple[float, str]:
    """Score with a pluggable judge.

    judge_fn may return either:
    - (normalized_score_0_to_1, reason)
    - a string like 'Score: 4\nReason: ...'
    - an integer/float score in 1-5 or 0-1 space
    """
    if judge_fn is None:
        return default_judge(question, expected, actual)

    raw = _call_judge(judge_fn, question, expected, actual)

    if isinstance(raw, tuple) and len(raw) == 2:
        score, reason = raw
        if score > 1.0:
            score = (float(score) - 1.0) / 4.0
        return max(0.0, min(1.0, float(score))), str(reason)

    if isinstance(raw, (int, float)):
        score = float(raw)
        if score > 1.0:
            score = (score - 1.0) / 4.0
        return max(0.0, min(1.0, score)), "Judge returned numeric score."

    text = str(raw).strip()
    score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    reason_match = re.search(r"Reason:\s*(.+)", text, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
        if score > 1.0:
            score = (score - 1.0) / 4.0
        return max(0.0, min(1.0, score)), (reason_match.group(1).strip() if reason_match else text)

    return default_judge(question, expected, actual)


def build_default_golden_dataset() -> List[GoldenTask]:
    """Default 15-task suite adapted to this codebase's integrated tools."""
    return [
        GoldenTask("math-001", "What is 347 * 23?", expected_answer="7981", expected_tools=["calculator"], category="math", difficulty="easy"),
        GoldenTask("math-002", "What is the square root of 1764?", expected_answer="42", expected_tools=["calculator"], category="math", difficulty="easy"),
        GoldenTask("math-003", "If a rectangle has area 120 and width 8, what is its length?", expected_answer="15", expected_tools=["calculator"], category="math", difficulty="medium"),
        GoldenTask("fact-001", "What is the capital of France?", expected_answer="Paris", expected_tools=["search_kb"], category="factual", difficulty="easy"),
        GoldenTask("fact-002", "What is the chemical symbol for gold?", expected_answer="Au", expected_tools=["search_kb"], category="factual", difficulty="easy"),
        GoldenTask("fact-003", "Who wrote the novel '1984'?", expected_answer="George Orwell", expected_tools=["search_kb"], category="factual", difficulty="easy"),
        GoldenTask("reason-001", "If all roses are flowers and some flowers are red, can we conclude all roses are red?", expected_answer="No", category="reasoning", difficulty="medium"),
        GoldenTask("reason-002", "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?", expected_answer="$0.05", expected_tools=["calculator"], category="reasoning", difficulty="medium"),
        GoldenTask("reason-003", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", expected_answer="5 minutes", expected_tools=["calculator"], category="reasoning", difficulty="hard"),
        GoldenTask("multi-001", "Calculate 15% tip on a $84.50 restaurant bill and round to the nearest cent.", expected_answer="$12.68", expected_tools=["calculator"], category="multi-step", difficulty="medium"),
        GoldenTask("multi-002", "Convert 72 degrees Fahrenheit to Celsius. Use the formula C = (F - 32) * 5/9.", expected_answer="22.22", expected_tools=["calculator"], category="multi-step", difficulty="medium"),
        GoldenTask("multi-003", "A store has a 25% off sale. An item costs $60 before discount. What is the sale price after 8% tax?", expected_answer="$48.60", expected_tools=["calculator"], category="multi-step", difficulty="hard"),
        GoldenTask("lookup-001", "Look up the population of Tokyo and report whether it exceeds 10 million.", expected_answer="Yes", expected_contains=["yes"], expected_tools=["search_kb"], category="lookup", difficulty="medium"),
        GoldenTask("lookup-002", "What is the boiling point of water in Fahrenheit?", expected_answer="212", expected_tools=["search_kb"], category="lookup", difficulty="easy"),
        GoldenTask("lookup-003", "How many planets are in our solar system?", expected_answer="8", expected_tools=["search_kb"], category="factual", difficulty="easy"),
    ]


def _extract_tool_names(tool_calls: Iterable[Any]) -> List[str]:
    names: List[str] = []
    for item in tool_calls or []:
        if isinstance(item, dict):
            name = item.get("tool") or item.get("tool_name")
        else:
            name = getattr(item, "tool", None) or getattr(item, "tool_name", None)
        if name is not None:
            names.append(str(name))
    return names


def _extract_token_counts(result: Any) -> Tuple[int, int, int]:
    metadata = getattr(result, "metadata", {}) or {}
    artifacts = getattr(result, "artifacts", {}) or {}
    raw = artifacts.get("raw", {}) if isinstance(artifacts, dict) else {}

    input_tokens = int(metadata.get("input_tokens", raw.get("input_tokens", 0)) or 0)
    output_tokens = int(metadata.get("output_tokens", raw.get("output_tokens", 0)) or 0)
    total_tokens = int(metadata.get("total_tokens", raw.get("total_tokens", input_tokens + output_tokens)) or (input_tokens + output_tokens))
    return input_tokens, output_tokens, total_tokens


def _extract_num_steps(result: Any) -> int:
    steps = getattr(result, "steps", None)
    if isinstance(steps, list):
        return len(steps)
    if steps is None:
        metadata = getattr(result, "metadata", {}) or {}
        return int(metadata.get("steps", metadata.get("num_steps", 0)) or 0)
    return int(steps)


def _extract_latency_seconds(result: Any, fallback_started_at: float) -> float:
    metadata = getattr(result, "metadata", {}) or {}
    if "duration_seconds" in metadata:
        return float(metadata["duration_seconds"])
    started_at = getattr(result, "started_at", None) or fallback_started_at
    finished_at = getattr(result, "finished_at", None) or time.time()
    return max(0.0, float(finished_at - started_at))


class AgentEvaluator:
    """Run an integrated runtime against a golden dataset and score it."""

    def __init__(self, runtime, name: str = "AgentRuntime", judge_fn: Optional[JudgeFn] = None):
        self.runtime = runtime
        self.name = name
        self.judge_fn = judge_fn
        self.results: List[EvalMetrics] = []

    def evaluate_single(self, test_case: GoldenTask, use_judge: bool = True) -> EvalMetrics:
        metrics = EvalMetrics(
            task_id=test_case.id,
            task_description=test_case.task,
            expected_answer=test_case.expected_answer,
            actual_answer="",
            category=test_case.category,
            difficulty=test_case.difficulty,
            tools_expected=list(test_case.expected_tools),
        )

        started_at = time.time()
        try:
            result = self.runtime.run(AgentRequest(
                task=test_case.task,
                strategy=test_case.strategy,
                max_steps=test_case.max_steps,
                metadata={"eval_task_id": test_case.id, **test_case.metadata},
            ))
            metrics.actual_answer = getattr(result, "answer", "") or ""
            metrics.strategy_used = getattr(result, "strategy_used", "unknown") or "unknown"
            metrics.request_id = getattr(result, "request_id", None)
            metrics.tools_used = _extract_tool_names(getattr(result, "tool_calls", []))
            metrics.num_steps = _extract_num_steps(result)
            metrics.input_tokens, metrics.output_tokens, metrics.total_tokens = _extract_token_counts(result)
            metrics.completed = True
            metrics.success = bool(getattr(result, "success", False))
            metrics.latency_seconds = _extract_latency_seconds(result, started_at)
            if not metrics.success:
                errors = getattr(result, "errors", []) or []
                if errors:
                    metrics.reasons.extend(str(e) for e in errors)
                else:
                    metrics.reasons.append("Agent returned unsuccessful result.")
        except Exception as e:
            metrics.completed = False
            metrics.success = False
            metrics.error = f"{type(e).__name__}: {e}"
            metrics.reasons.append(metrics.error)
            metrics.latency_seconds = time.time() - started_at
            return metrics

        if test_case.expected_answer:
            metrics.exact_match = score_exact_match(test_case.expected_answer, metrics.actual_answer)
            metrics.fuzzy_score = score_fuzzy_match(test_case.expected_answer, metrics.actual_answer)
            if use_judge:
                judge_expected = test_case.expected_answer
                metrics.judge_score, judge_reason = score_with_judge(
                    test_case.task,
                    judge_expected,
                    metrics.actual_answer,
                    judge_fn=self.judge_fn,
                )
                if judge_reason and metrics.judge_score < 1.0 and metrics.exact_match < 1.0:
                    metrics.reasons.append(f"Judge: {judge_reason}")
            else:
                metrics.judge_score = metrics.fuzzy_score

            answer_ok = metrics.exact_match == 1.0 or metrics.judge_score >= 0.95
            if not answer_ok:
                metrics.reasons.append(f"Expected answer '{test_case.expected_answer}' but got '{metrics.actual_answer}'.")

        metrics.contains_score, contains_failures = score_contains_all(test_case.expected_contains, metrics.actual_answer)
        metrics.forbidden_score, forbidden_failures = score_forbidden_absent(test_case.forbidden_contains, metrics.actual_answer)
        metrics.reasons.extend(contains_failures)
        metrics.reasons.extend(forbidden_failures)

        if test_case.expected_tools or metrics.tools_used:
            metrics.tool_accuracy = score_tool_accuracy(test_case.expected_tools, metrics.tools_used)
            for tool in test_case.expected_tools:
                if tool not in metrics.tools_used:
                    metrics.reasons.append(f"Expected tool not called: {tool}")
        else:
            metrics.tool_accuracy = 1.0

        # If the task is successful on all constraints, clear purely judge-level reasons.
        if (
            metrics.success
            and (not test_case.expected_answer or metrics.exact_match == 1.0)
            and metrics.contains_score == 1.0
            and metrics.forbidden_score == 1.0
            and metrics.tool_accuracy == 1.0
        ):
            metrics.reasons = [r for r in metrics.reasons if not r.startswith("Judge:")]

        return metrics

    def evaluate_dataset(self, dataset: List[GoldenTask], use_judge: bool = True) -> List[EvalMetrics]:
        self.results = []
        for test_case in dataset:
            self.results.append(self.evaluate_single(test_case, use_judge=use_judge))
        return self.results

    def aggregate_report(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No results to aggregate", "agent_name": self.name}

        completed = [r for r in self.results if r.completed]
        passed = [r for r in self.results if r.passed]
        n = len(self.results)

        report = {
            "agent_name": self.name,
            "total_tasks": n,
            "completed": len(completed),
            "passed": len(passed),
            "completion_rate": len(completed) / n if n else 0.0,
            "pass_rate": len(passed) / n if n else 0.0,
            "avg_exact_match": sum(r.exact_match for r in completed) / max(len(completed), 1),
            "avg_fuzzy_score": sum(r.fuzzy_score for r in completed) / max(len(completed), 1),
            "avg_judge_score": sum(r.judge_score for r in completed) / max(len(completed), 1),
            "avg_tool_accuracy": sum(r.tool_accuracy for r in completed) / max(len(completed), 1),
            "avg_composite": sum(r.composite_score for r in completed) / max(len(completed), 1),
            "total_tokens": sum(r.total_tokens for r in completed),
            "avg_tokens_per_task": sum(r.total_tokens for r in completed) / max(len(completed), 1),
            "avg_latency": sum(r.latency_seconds for r in completed) / max(len(completed), 1),
            "avg_steps": sum(r.num_steps for r in completed) / max(len(completed), 1),
            "failures": [
                {
                    "task_id": r.task_id,
                    "reasons": r.reasons,
                    "answer": r.actual_answer,
                    "strategy_used": r.strategy_used,
                }
                for r in self.results if not r.passed
            ],
            "results": [r.to_dict() for r in self.results],
        }

        by_category: Dict[str, List[EvalMetrics]] = defaultdict(list)
        by_difficulty: Dict[str, List[EvalMetrics]] = defaultdict(list)
        for item in completed:
            by_category[item.category].append(item)
            by_difficulty[item.difficulty].append(item)

        report["by_category"] = {
            category: {
                "count": len(items),
                "avg_composite": sum(r.composite_score for r in items) / len(items),
                "avg_exact_match": sum(r.exact_match for r in items) / len(items),
                "pass_rate": sum(1 for r in items if r.passed) / len(items),
            }
            for category, items in by_category.items()
        }
        report["by_difficulty"] = {
            difficulty: {
                "count": len(items),
                "avg_composite": sum(r.composite_score for r in items) / len(items),
                "pass_rate": sum(1 for r in items if r.passed) / len(items),
            }
            for difficulty, items in by_difficulty.items()
        }
        return report

    def print_report(self) -> None:
        report = self.aggregate_report()
        if report.get("error"):
            print(report["error"])
            return
        print(f"\n{'=' * 60}")
        print(f"  EVALUATION REPORT: {report['agent_name']}")
        print(f"{'=' * 60}")
        print(f"  Tasks: {report['passed']}/{report['total_tasks']} passed ({report['pass_rate']:.0%})")
        print(f"  Completed: {report['completed']}/{report['total_tasks']} ({report['completion_rate']:.0%})")
        print("\n  Scores:")
        print(f"    Exact Match:    {report['avg_exact_match']:.2%}")
        print(f"    Fuzzy Match:    {report['avg_fuzzy_score']:.2%}")
        print(f"    Judge:          {report['avg_judge_score']:.2%}")
        print(f"    Tool Accuracy:  {report['avg_tool_accuracy']:.2%}")
        print(f"    Composite:      {report['avg_composite']:.2%}")
        print("\n  Cost:")
        print(f"    Total tokens:   {report['total_tokens']:,}")
        print(f"    Avg/task:       {report['avg_tokens_per_task']:.0f}")
        print(f"    Avg latency:    {report['avg_latency']:.2f}s")
        print(f"    Avg steps:      {report['avg_steps']:.1f}")
        if report["by_category"]:
            print("\n  By Category:")
            for category, stats in sorted(report["by_category"].items()):
                print(f"    {category}: {stats['pass_rate']:.0%} pass, {stats['avg_composite']:.2%} composite ({stats['count']} tasks)")
        print(f"{'=' * 60}")


class CostTracker:
    """Estimate evaluation cost from EvalMetrics."""

    COST_PER_1M_INPUT_TOKENS = 0.15
    COST_PER_1M_OUTPUT_TOKENS = 0.60

    def __init__(self):
        self.tasks: List[Dict[str, Any]] = []

    def record(self, metrics: EvalMetrics) -> None:
        estimated_cost = (
            metrics.input_tokens * self.COST_PER_1M_INPUT_TOKENS / 1_000_000 +
            metrics.output_tokens * self.COST_PER_1M_OUTPUT_TOKENS / 1_000_000
        )
        self.tasks.append({
            "task_id": metrics.task_id,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "total_tokens": metrics.total_tokens,
            "latency": metrics.latency_seconds,
            "steps": metrics.num_steps,
            "estimated_cost": estimated_cost,
        })

    def extend(self, results: Iterable[EvalMetrics]) -> None:
        for metrics in results:
            self.record(metrics)

    def summary(self) -> Dict[str, Any]:
        if not self.tasks:
            return {
                "num_tasks": 0,
                "total_tokens": 0,
                "avg_tokens_per_task": 0.0,
                "total_estimated_cost": 0.0,
                "avg_cost_per_task": 0.0,
                "total_time": 0.0,
                "avg_latency": 0.0,
                "tokens_per_second": 0.0,
            }
        total_input = sum(t["input_tokens"] for t in self.tasks)
        total_output = sum(t["output_tokens"] for t in self.tasks)
        total_cost = sum(t["estimated_cost"] for t in self.tasks)
        total_time = sum(t["latency"] for t in self.tasks)
        n = len(self.tasks)
        return {
            "num_tasks": n,
            "total_tokens": total_input + total_output,
            "avg_tokens_per_task": (total_input + total_output) / n,
            "total_estimated_cost": total_cost,
            "avg_cost_per_task": total_cost / n,
            "total_time": total_time,
            "avg_latency": total_time / n,
            "tokens_per_second": (total_input + total_output) / max(total_time, 0.01),
        }


class RegressionTracker:
    """Save evaluation baselines and compare later runs."""

    def __init__(self):
        self.baselines: Dict[str, Dict[str, Any]] = {}

    def save_baseline(self, name: str, report: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "report": report,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.baselines[name] = entry
        return entry

    def save_baseline_file(self, path: str, report: Dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "report": report,
        }, indent=2), encoding="utf-8")

    def load_baseline_file(self, path: str, name: Optional[str] = None) -> Dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        baseline_name = name or Path(path).stem
        self.baselines[baseline_name] = payload
        return payload

    def compare(self, name: str, new_report: Dict[str, Any], threshold: float = 0.05) -> Dict[str, Any]:
        if name not in self.baselines:
            return {"error": f"No baseline named '{name}'"}

        baseline = self.baselines[name]["report"]
        comparison = {"baseline_name": name, "metrics": {}, "regressions": []}
        metrics_to_compare = [
            "pass_rate",
            "avg_exact_match",
            "avg_fuzzy_score",
            "avg_judge_score",
            "avg_tool_accuracy",
            "avg_composite",
            "avg_tokens_per_task",
            "avg_latency",
        ]

        for metric in metrics_to_compare:
            old_val = float(baseline.get(metric, 0.0) or 0.0)
            new_val = float(new_report.get(metric, 0.0) or 0.0)
            delta = new_val - old_val
            is_cost_metric = metric in {"avg_tokens_per_task", "avg_latency"}
            improved = delta < -threshold if is_cost_metric else delta > threshold
            regressed = delta > threshold if is_cost_metric else delta < -threshold
            comparison["metrics"][metric] = {
                "baseline": old_val,
                "current": new_val,
                "delta": delta,
                "status": "improved" if improved else ("regressed" if regressed else "stable"),
            }
            if regressed:
                comparison["regressions"].append(metric)
        return comparison


class RegressionRunner:
    """Backwards-compatible wrapper returning simple pass/fail suite summaries."""

    def __init__(self, runtime, judge_fn: Optional[JudgeFn] = None, name: str = "AgentRuntime"):
        self.evaluator = AgentEvaluator(runtime, name=name, judge_fn=judge_fn)

    def run_task(self, task: GoldenTask) -> EvalResult:
        metrics = self.evaluator.evaluate_single(task, use_judge=True)
        return EvalResult(
            task_id=metrics.task_id,
            passed=metrics.passed,
            score=metrics.composite_score,
            reasons=metrics.reasons,
            answer=metrics.actual_answer,
            strategy_used=metrics.strategy_used,
            tools_used=metrics.tools_used,
        )

    def run_suite(self, tasks: List[GoldenTask]) -> Dict[str, Any]:
        metrics_list = self.evaluator.evaluate_dataset(tasks, use_judge=True)
        results = [
            EvalResult(
                task_id=m.task_id,
                passed=m.passed,
                score=m.composite_score,
                reasons=m.reasons,
                answer=m.actual_answer,
                strategy_used=m.strategy_used,
                tools_used=m.tools_used,
            ).to_dict()
            for m in metrics_list
        ]
        passed = sum(1 for item in results if item["passed"])
        return {
            "passed": passed,
            "total": len(results),
            "pass_rate": passed / len(results) if results else 0.0,
            "results": results,
            "report": self.evaluator.aggregate_report(),
        }


__all__ = [
    "GoldenTask",
    "EvalMetrics",
    "EvalResult",
    "AgentEvaluator",
    "CostTracker",
    "RegressionTracker",
    "RegressionRunner",
    "build_default_golden_dataset",
    "normalize_answer",
    "score_exact_match",
    "score_fuzzy_match",
    "score_tool_accuracy",
    "score_with_judge",
    "default_judge",
]
