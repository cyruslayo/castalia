"""Cost/latency optimization primitives for Notebook 27 foundation.

This module lifts the teaching components from Notebook 27 into reusable,
deterministic runtime helpers that can be tested without a live LLM.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used for budgeting before exact provider usage exists."""
    text = str(text or "")
    if not text.strip():
        return 0
    # Keep the heuristic simple and stable for tests.
    return max(1, int(len(text.split()) * 1.3))


class ResponseCache:
    """Cache responses by a deterministic hash of payload + generation params."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._store: Dict[str, dict] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, namespace: str, payload: Any, **kwargs) -> str:
        raw = json.dumps({"payload": payload, "params": kwargs}, sort_keys=True, default=str)
        return namespace + ":" + hashlib.sha256(raw.encode()).hexdigest()

    def get(self, namespace: str, payload: Any, **kwargs) -> Optional[Any]:
        entry = self._store.get(self._key(namespace, payload, **kwargs))
        if entry is None:
            self.misses += 1
            return None
        entry["hits"] += 1
        self.hits += 1
        return entry["value"]

    def set(self, namespace: str, payload: Any, value: Any, **kwargs) -> None:
        if len(self._store) >= self.max_size:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[self._key(namespace, payload, **kwargs)] = {
            "value": value,
            "created": time.time(),
            "hits": 0,
        }

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
        }


# Backward-compatible alias from the earlier placeholder implementation.
SimpleCache = ResponseCache


@dataclass
class TokenBudget:
    """Track cumulative token usage with threshold alerts and hard limits."""

    max_tokens: int = 50_000
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    calls: List[dict] = field(default_factory=list)
    alerts_fired: set = field(default_factory=set)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    warnings: List[str] = field(default_factory=list)
    max_context_tokens: int = 8192
    reserved_response_tokens: int = 2048

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def utilization(self) -> float:
        return self.total_tokens / self.max_tokens if self.max_tokens > 0 else 0.0

    def record_call(self, input_text: str, output_text: str, label: str = "") -> dict:
        input_tokens = estimate_tokens(input_text)
        output_tokens = estimate_tokens(output_text)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        record = {
            "label": label,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cumulative": self.total_tokens,
            "utilization": self.utilization,
        }
        self.calls.append(record)
        for threshold in self.alert_thresholds:
            if self.utilization >= threshold and threshold not in self.alerts_fired:
                self.alerts_fired.add(threshold)
                self.warnings.append(
                    f"Budget alert: {threshold:.0%} used ({self.total_tokens}/{self.max_tokens} tokens)"
                )
        return record

    def is_over_budget(self) -> bool:
        return self.total_tokens >= self.max_tokens

    def check(self, messages: List[dict]) -> dict:
        used = sum(estimate_tokens(m.get("content", "")) for m in messages)
        available = self.max_context_tokens - self.reserved_response_tokens
        ok = used <= available
        if not ok:
            self.warnings.append(f"Estimated prompt tokens {used} exceed budget {available}")
        return {"ok": ok, "estimated_tokens": used, "budget": available}

    def summary(self) -> str:
        return (
            f"Budget: {self.total_tokens:,}/{self.max_tokens:,} tokens "
            f"({self.utilization:.1%}) | Calls: {len(self.calls)} | "
            f"Input: {self.total_input_tokens:,} | Output: {self.total_output_tokens:,}"
        )


class PromptCompressor:
    """Compress long chat histories by summarizing older turns."""

    def __init__(self, max_history_tokens: int = 1500, summary_trigger: int = 3):
        self.max_history_tokens = max_history_tokens
        self.summary_trigger = summary_trigger
        self.compressions = 0
        self.tokens_saved = 0

    def estimate_tokens(self, messages: List[dict]) -> int:
        return sum(estimate_tokens(m.get("content", "")) for m in messages)

    def compress(self, messages: List[dict]) -> List[dict]:
        if len(messages) < self.summary_trigger + 2:
            return messages
        current_tokens = self.estimate_tokens(messages)
        if current_tokens <= self.max_history_tokens:
            return messages

        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        start_idx = 1 if system_msg else 0
        middle = messages[start_idx:-2]
        recent = messages[-2:]
        if not middle:
            return messages

        summary_parts = []
        for msg in middle:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "assistant":
                thought = re.search(r"Thought:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
                action = re.search(r"Action:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
                final_answer = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
                if thought and action:
                    summary_parts.append(f"Thought: {thought.group(1)[:80]} -> Action: {action.group(1)[:80]}")
                elif final_answer:
                    summary_parts.append(f"Final: {final_answer.group(1)[:100]}")
                else:
                    summary_parts.append(f"Assistant: {content[:100]}")
            else:
                prefix = "Observation" if content.startswith("Observation:") else "User"
                summary_parts.append(f"{prefix}: {content[:100]}")

        summary_text = "Previous conversation summary:\n" + "\n".join(summary_parts)
        compressed: List[dict] = []
        if system_msg:
            compressed.append(system_msg)
        compressed.append({"role": "user", "content": summary_text})
        compressed.extend(recent)

        new_tokens = self.estimate_tokens(compressed)
        saved = current_tokens - new_tokens
        self.compressions += 1
        if saved > 0:
            self.tokens_saved += saved
        return compressed

    def stats(self) -> dict:
        return {"compressions": self.compressions, "tokens_saved": self.tokens_saved}


class ModelRouter:
    """Route tasks to generation profiles based on cheap heuristics."""

    PROFILES = {
        "simple": {
            "max_new_tokens": 128,
            "temperature": 0.1,
            "do_sample": True,
        },
        "moderate": {
            "max_new_tokens": 256,
            "temperature": 0.5,
            "do_sample": True,
        },
        "complex": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
        },
    }

    SIMPLE_SIGNALS = ["what is", "capital of", "symbol for", "how many", "boiling point"]
    COMPLEX_SIGNALS = ["if .* then", "compare", "explain why", "step by step", "after .* what"]

    def __init__(self, default_model: Optional[str] = None):
        self.default_model = default_model
        self.routing_log: List[dict] = []

    def classify_difficulty(self, query: str) -> str:
        query_lower = str(query or "").lower()
        for pattern in self.COMPLEX_SIGNALS:
            if re.search(pattern, query_lower):
                return "complex"
        for signal in self.SIMPLE_SIGNALS:
            if signal in query_lower:
                return "simple"
        return "moderate"

    def choose(self, task: str, complexity: str = "normal") -> str:
        return self.default_model or "default"

    def get_params(self, query: str) -> Dict[str, Any]:
        difficulty = self.classify_difficulty(query)
        profile = dict(self.PROFILES[difficulty])
        self.routing_log.append({
            "query": str(query)[:80],
            "difficulty": difficulty,
            "max_new_tokens": profile["max_new_tokens"],
            "model": self.choose(query, difficulty),
        })
        return profile

    def stats(self) -> dict:
        counts: Dict[str, int] = {"simple": 0, "moderate": 0, "complex": 0}
        for entry in self.routing_log:
            counts[entry["difficulty"]] = counts.get(entry["difficulty"], 0) + 1
        total_budget = sum(e["max_new_tokens"] for e in self.routing_log)
        return {
            "routed": len(self.routing_log),
            "by_difficulty": counts,
            "token_budget": total_budget,
        }


class CircularityDetector:
    """Detect repeated actions/responses and recommend early termination."""

    def __init__(self, window_size: int = 5, similarity_threshold: float = 0.8):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.action_history: List[str] = []
        self.circular_detected = False

    def _normalize_action(self, action: str) -> str:
        return re.sub(r"\s+", " ", str(action or "").lower().strip())

    def _similarity(self, a: str, b: str) -> float:
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        overlap = words_a & words_b
        return len(overlap) / max(len(words_a), len(words_b))

    def record_action(self, action: str) -> bool:
        normalized = self._normalize_action(action)
        self.action_history.append(normalized)
        if len(self.action_history) < 3:
            return False
        recent = self.action_history[-self.window_size:]
        duplicates = sum(
            1 for prev in recent[:-1]
            if self._similarity(prev, recent[-1]) >= self.similarity_threshold
        )
        if duplicates >= 2:
            self.circular_detected = True
            return True
        return False

    def reset(self) -> None:
        self.action_history = []
        self.circular_detected = False


def generate_cached(
    namespace: str,
    payload: Any,
    cache: ResponseCache,
    generator,
    **kwargs,
) -> Tuple[Any, bool]:
    cached = cache.get(namespace, payload, **kwargs)
    if cached is not None:
        return cached, True
    value = generator(payload, **kwargs)
    cache.set(namespace, payload, value, **kwargs)
    return value, False


class RuntimeOptimizer:
    """Bundle Notebook 27 primitives for runtime use."""

    def __init__(
        self,
        *,
        max_tokens: int = 50_000,
        max_history_tokens: int = 1_500,
        summary_trigger: int = 3,
        default_model: Optional[str] = None,
        cache_size: int = 1_000,
        context_window: int = 8_192,
        reserved_response_tokens: int = 2_048,
    ):
        self.budget = TokenBudget(
            max_tokens=max_tokens,
            max_context_tokens=context_window,
            reserved_response_tokens=reserved_response_tokens,
        )
        self.cache = ResponseCache(max_size=cache_size)
        self.compressor = PromptCompressor(
            max_history_tokens=max_history_tokens,
            summary_trigger=summary_trigger,
        )
        self.router = ModelRouter(default_model=default_model)
        self.circularity = CircularityDetector()

    def reset_run(self) -> None:
        self.circularity.reset()

    def route_for_query(self, query: str) -> Dict[str, Any]:
        return self.router.get_params(query)

    def prepare_messages(self, messages: List[dict]) -> Tuple[List[dict], dict]:
        compressed = self.compressor.compress(messages)
        budget_check = self.budget.check(compressed)
        return compressed, budget_check

    def cached_generate(self, namespace: str, payload: Any, generator, **kwargs) -> Tuple[Any, bool]:
        return generate_cached(namespace, payload, self.cache, generator, **kwargs)

    def record_generation(self, messages: List[dict], response: str, *, label: str = "", cached: bool = False) -> None:
        if cached:
            return
        input_text = " ".join(str(m.get("content", "")) for m in messages)
        self.budget.record_call(input_text, response, label=label)

    def detect_circularity(self, text: str) -> bool:
        return self.circularity.record_action(text)

    def stats(self) -> dict:
        return {
            "budget": self.budget.summary(),
            "cache": self.cache.stats(),
            "compression": self.compressor.stats(),
            "routing": self.router.stats(),
            "circularity": {
                "detected": self.circularity.circular_detected,
                "history_size": len(self.circularity.action_history),
            },
            "warnings": list(self.budget.warnings),
        }
