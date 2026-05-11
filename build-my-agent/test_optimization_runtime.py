from optimization_runtime import (
    CircularityDetector,
    ModelRouter,
    PromptCompressor,
    ResponseCache,
    TokenBudget,
    estimate_tokens,
    generate_cached,
)


def test_estimate_tokens_and_budget_tracking():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") >= 2

    budget = TokenBudget(max_tokens=20, alert_thresholds=[0.25, 0.5])
    rec = budget.record_call("one two three four", "alpha beta", label="step-1")

    assert rec["label"] == "step-1"
    assert budget.total_tokens == rec["input_tokens"] + rec["output_tokens"]
    assert budget.utilization > 0
    assert any("25%" in warning or "50%" in warning for warning in budget.warnings)


def test_token_budget_context_check_flags_overflow():
    budget = TokenBudget(max_context_tokens=10, reserved_response_tokens=2)
    result = budget.check([
        {"role": "user", "content": "one two three four five six seven eight nine ten"},
    ])
    assert result["ok"] is False
    assert result["budget"] == 8
    assert budget.warnings


def test_response_cache_and_generate_cached():
    cache = ResponseCache(max_size=2)
    calls = {"count": 0}

    def fake_generate(payload, **kwargs):
        calls["count"] += 1
        return f"answer:{payload['q']}:{kwargs['temperature']}"

    payload = {"q": "What is 2+2?"}
    out1, cached1 = generate_cached("llm", payload, cache, fake_generate, temperature=0.1)
    out2, cached2 = generate_cached("llm", payload, cache, fake_generate, temperature=0.1)

    assert cached1 is False
    assert cached2 is True
    assert out1 == out2
    assert calls["count"] == 1
    assert cache.stats()["hit_rate"] == 0.5


def test_prompt_compressor_reduces_history_and_preserves_recent_turns():
    compressor = PromptCompressor(max_history_tokens=20, summary_trigger=2)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Question one with extra words for length."},
        {"role": "assistant", "content": "Thought: search first\nAction: search_kb"},
        {"role": "user", "content": "Observation: Paris is the capital of France."},
        {"role": "assistant", "content": "Thought: answer now\nFinal Answer: Paris"},
        {"role": "user", "content": "Thanks now do another task."},
    ]

    compressed = compressor.compress(messages)
    assert len(compressed) < len(messages)
    assert compressed[0]["role"] == "system"
    assert "Previous conversation summary" in compressed[1]["content"]
    assert compressed[-2:] == messages[-2:]
    assert compressor.stats()["compressions"] == 1


def test_model_router_classifies_and_tracks_profiles():
    router = ModelRouter(default_model="test-model")
    simple = router.get_params("What is the capital of France?")
    complex_params = router.get_params("If a store gives 25% off then adds tax, what happens?")
    moderate = router.get_params("Summarize this article")

    assert simple["max_new_tokens"] == 128
    assert complex_params["max_new_tokens"] == 512
    assert moderate["max_new_tokens"] == 256
    stats = router.stats()
    assert stats["routed"] == 3
    assert stats["by_difficulty"]["simple"] == 1
    assert stats["by_difficulty"]["complex"] == 1


def test_circularity_detector_flags_repetition_and_resets():
    detector = CircularityDetector(window_size=5, similarity_threshold=0.8)
    assert detector.record_action("Action: calculator input: 347 * 23") is False
    assert detector.record_action("Action: calculator input: 347 * 23") is False
    assert detector.record_action("Action: calculator input: 347 * 23") is True
    assert detector.circular_detected is True
    detector.reset()
    assert detector.circular_detected is False
    assert detector.action_history == []
