"""
Memory - Agent Short-Term Memory Management (Notebook 10)

Four strategies for managing conversation history as it grows:
  1. FullHistory   -- keep everything (perfect recall, unbounded cost)
  2. SlidingWindow -- keep last N messages (bounded, hard cutoff loss)
  3. Summarization -- compress old messages (saves tokens, lossy)
  4. Importance    -- score & keep what matters (smart, heuristic-dependent)

Plus a unified MemoryManager controller that swaps strategies without
changing agent code.

Backward-compatible: build_prompt() and trim_for_api() still work so
agent_loop.py does not break.
"""

import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# ==============================================================================
# SECTION 1 -- Message dataclass (typed conversation unit)
# ==============================================================================
#
# Design choice: we use a dataclass instead of raw dicts so the compiler
# catches field-name typos and IDE gives us autocomplete.
# The _token_count field is lazy -- we only compute it when asked, because
# tokenisation is expensive and we may never need the number.

@dataclass
class Message:
    """A single conversation message with optional metadata."""
    role: str                       # "system" | "user" | "assistant"
    content: str                    # The actual text
    timestamp: float = field(default_factory=time.time)
    _token_count: Optional[int] = None  # Lazy -- filled on first estimate

    def to_dict(self) -> Dict[str, str]:
        """Convert back to the OpenAI API message format."""
        return {"role": self.role, "content": self.content}

    def estimate_tokens(self) -> int:
        """
        Rough token count: ~4 characters per token for English text.
        This is fast and good-enough for memory management decisions.
        For production you would use tiktoken or the model's tokenizer.
        """
        if self._token_count is None:
            self._token_count = max(1, len(self.content) // 4)
        return self._token_count


# ==============================================================================
# SECTION 2 -- Abstract base for all memory strategies
# ==============================================================================
#
# Design choice: Strategy pattern (GoF).  Each concrete strategy implements
# the same interface so MemoryManager can swap them at runtime without
# touching agent code.

class MemoryStrategy(ABC):
    """Abstract base class -- every strategy must implement these."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Add a message to memory."""

    @abstractmethod
    def get_context(self) -> List[Message]:
        """Return the messages that should be sent to the LLM."""

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return diagnostic info (message count, token estimate, etc.)."""


# ==============================================================================
# SECTION 3 -- Strategy 1: FullHistory
# ==============================================================================
#
# PROS:  Perfect recall -- nothing is ever lost.
# CONS:  Token explosion.  At ~200 tokens/message, 300 steps = 60K tokens
#        which exceeds most context windows and costs real money.
# WHEN:  Only for debugging or very short conversations (<20 steps).

class FullHistoryMemory(MemoryStrategy):
    """Keep every message.  No trimming, no summarisation."""

    def __init__(self):
        self._messages: List[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def get_context(self) -> List[Message]:
        return list(self._messages)  # return a copy so callers cannot mutate

    def stats(self) -> Dict[str, Any]:
        total_tokens = sum(m.estimate_tokens() for m in self._messages)
        return {
            "strategy": "full_history",
            "message_count": len(self._messages),
            "estimated_tokens": total_tokens,
        }


# ==============================================================================
# SECTION 4 -- Strategy 2: SlidingWindow
# ==============================================================================
#
# PROS:  Predictable token budget.  O(1) add, O(N) trim.
# CONS:  Hard cutoff -- loses important early context.  If the system
#        prompt or the original goal falls out of the window, the agent
#        literally forgets who it is and what it was asked to do.
# WHEN:  Good default for most single-task agents.

class SlidingWindowMemory(MemoryStrategy):
    """Keep only the most recent N messages."""

    def __init__(self, window_size: int = 10):
        if window_size < 2:
            raise ValueError("window_size must be >= 2 (need at least one Q+A pair)")
        self._window_size = window_size
        self._messages: List[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        # Evict oldest messages when we exceed the window
        if len(self._messages) > self._window_size:
            self._messages = self._messages[-self._window_size:]

    def get_context(self) -> List[Message]:
        return list(self._messages)

    def stats(self) -> Dict[str, Any]:
        total_tokens = sum(m.estimate_tokens() for m in self._messages)
        return {
            "strategy": "sliding_window",
            "window_size": self._window_size,
            "message_count": len(self._messages),
            "estimated_tokens": total_tokens,
        }


# ==============================================================================
# SECTION 5 -- Strategy 3: SummarizingMemory
# ==============================================================================
#
# PROS:  Preserves key information at lower cost.  Scales to very long
#        conversations because old context is compressed.
# CONS:  Summarisation is LOSSY -- details are inevitably dropped.
#        If you use an LLM to summarise, each compression costs an
#        additional API call (more money, more latency).
# WHEN:  Long multi-turn conversations where some forgetting is acceptable.
#
# Implementation note: We provide TWO summarisation modes:
#   (a) heuristic -- fast, no LLM needed, just extract key sentences
#   (b) llm       -- calls an LLM to write a summary (more accurate, more cost)
# The default is heuristic so this module works without an API key.

class SummarizingMemory(MemoryStrategy):
    """Compress old messages into a running summary, keep recent raw."""

    def __init__(
        self,
        summary_threshold: int = 5,
        keep_recent: int = 2,
        mode: str = "heuristic",
        llm_fn=None,
    ):
        """
        Args:
            summary_threshold: Compress every N messages added.
            keep_recent:       How many of the newest messages to keep raw.
            mode:              "heuristic" (fast, no LLM) or "llm" (calls LLM).
            llm_fn:            Callable(text) -> str, only used when mode=="llm".
        """
        self._summary_threshold = summary_threshold
        self._keep_recent = keep_recent
        self._mode = mode
        self._llm_fn = llm_fn
        self._summary: str = ""
        self._recent: List[Message] = []
        self._buffer: List[Message] = []        # staging area
        self._compress_count: int = 0            # how many times we compressed

    # ----- public interface ---------------------------------------------------

    def add(self, message: Message) -> None:
        self._buffer.append(message)
        if len(self._buffer) >= self._summary_threshold:
            self._compress()

    def get_context(self) -> List[Message]:
        """Build the final prompt: [summary] + [recent messages] + [buffer]."""
        result = []
        if self._summary:
            result.append(Message(role="user", content=f"[MEMORY SUMMARY]\n{self._summary}"))
        result.extend(self._recent)
        # Include buffer messages (not yet compressed) so nothing is lost
        result.extend(self._buffer)
        return result

    def stats(self) -> Dict[str, Any]:
        total_tokens = (
            len(self._summary) // 4
            + sum(m.estimate_tokens() for m in self._recent)
            + sum(m.estimate_tokens() for m in self._buffer)
        )
        return {
            "strategy": "summarizing",
            "summary_length": len(self._summary),
            "recent_count": len(self._recent),
            "buffer_count": len(self._buffer),
            "compressions": self._compress_count,
            "estimated_tokens": total_tokens,
        }

    # ----- internal -----------------------------------------------------------

    def _compress(self) -> None:
        """Compress the buffer into the running summary."""
        if self._mode == "llm" and self._llm_fn is not None:
            # Call an external LLM to summarise (more accurate, more cost)
            text = "\n".join(m.content for m in self._buffer)
            new_summary = self._llm_fn(text)
        else:
            # Heuristic: extract key sentences (fast, free, lossy)
            new_summary = self._heuristic_summary(self._buffer)

        # Append to running summary (with a separator so old vs new is clear)
        if self._summary:
            self._summary += "\n---\n" + new_summary
        else:
            self._summary = new_summary

        # Move the most recent messages out of the buffer so they stay raw
        self._recent.extend(self._buffer[-self._keep_recent:])
        self._buffer = []
        self._compress_count += 1

    @staticmethod
    def _heuristic_summary(messages: List[Message]) -> str:
        """
        Fast, no-LLM summarisation.
        Strategy: extract the first sentence of each assistant message,
        and note how many user messages were in the batch.
        """
        lines = []
        user_count = 0
        for msg in messages:
            if msg.role == "user":
                user_count += 1
            elif msg.role == "assistant":
                # Take first 80 chars as a rough "key sentence"
                first_line = msg.content.split("\n")[0][:80]
                if first_line:
                    lines.append(f"  Agent: {first_line}")

        parts = [f"(compressed {len(messages)} messages, {user_count} from user)"]
        parts.extend(lines)
        return "\n".join(parts)


# ==============================================================================
# SECTION 6 -- Strategy 4: ImportanceWeightedMemory
# ==============================================================================
#
# PROS:  Smart retention -- keeps what matters, discards noise.
#        No arbitrary cutoff (unlike sliding window).
# CONS:  The importance heuristic is imperfect and domain-specific.
#        What is "important" for a math agent differs from a creative
#        writing agent.
# WHEN:  Conversations where some turns are genuinely more valuable
#        than others (e.g., research, debugging, multi-step reasoning).
#
# Scoring factors:
#   - Recency:     newer messages get a bonus (they are fresher in context)
#   - Length:      longer messages often contain more information
#   - Keywords:    words like "therefore", "conclusion", "result" signal
#                  important content
#   - Role:        assistant answers > observations > questions
#   - Tool calls:  messages that executed tools are inherently important

class ImportanceWeightedMemory(MemoryStrategy):
    """Score each message by importance, keep high-score ones."""

    # Keywords that signal important content
    IMPORTANCE_KEYWORDS = [
        "therefore", "conclusion", "important", "key",
        "answer", "result", "discovered", "found",
        "solution", "final", "determined", "calculated",
        "error", "failed", "exception", "critical",
    ]

    def __init__(
        self,
        importance_threshold: float = 0.5,
        max_messages: int = 50,
    ):
        """
        Args:
            importance_threshold: Minimum score (0.0-1.0) to keep a message.
            max_messages:         Hard ceiling to prevent unbounded growth.
        """
        self._threshold = importance_threshold
        self._max_messages = max_messages
        self._messages: List[Tuple[Message, float]] = []  # (msg, score)

    def add(self, message: Message) -> None:
        score = self._score_message(message)
        self._messages.append((message, score))
        # Evict if we exceed the hard ceiling
        if len(self._messages) > self._max_messages:
            self._evict_lowest()

    def get_context(self) -> List[Message]:
        """Return messages above the importance threshold, sorted by time."""
        kept = [msg for msg, score in self._messages if score >= self._threshold]
        # Always sort chronologically so the LLM sees a coherent conversation
        kept.sort(key=lambda m: m.timestamp)
        return kept

    def stats(self) -> Dict[str, Any]:
        total_tokens = sum(m.estimate_tokens() for m, _ in self._messages)
        kept_tokens = sum(m.estimate_tokens() for m in self.get_context())
        return {
            "strategy": "importance_weighted",
            "total_messages": len(self._messages),
            "kept_messages": len(self.get_context()),
            "total_tokens": total_tokens,
            "kept_tokens": kept_tokens,
            "threshold": self._threshold,
        }

    # ----- scoring ------------------------------------------------------------

    def _score_message(self, message: Message) -> float:
        """
        Calculate importance score for a single message.
        Returns a float between 0.0 and 1.0.

        The formula combines four factors:
          score = (recency * 0.3) + (length * 0.2) + (keyword * 0.3) + (role * 0.2)
        """
        recency = self._recency_score(message)
        length = self._length_score(message)
        keyword = self._keyword_score(message)
        role = self._role_score(message)

        return (recency * 0.3) + (length * 0.2) + (keyword * 0.3) + (role * 0.2)

    def _recency_score(self, message: Message) -> float:
        """Newer messages score higher (0.0 = old, 1.0 = just added)."""
        if not self._messages:
            return 1.0
        timestamps = [m.timestamp for m, _ in self._messages]
        timestamps.append(message.timestamp)
        t_min, t_max = min(timestamps), max(timestamps)
        if t_max == t_min:
            return 0.5
        return (message.timestamp - t_min) / (t_max - t_min)

    @staticmethod
    def _length_score(message: Message) -> float:
        """Longer messages tend to contain more information."""
        length = len(message.content)
        # Sigmoid-like curve: 0-50 chars = low, 200+ = high
        return min(1.0, length / 200.0)

    @staticmethod
    def _keyword_score(message: Message) -> float:
        """Messages with importance keywords score higher."""
        content_lower = message.content.lower()
        hits = sum(1 for kw in ImportanceWeightedMemory.IMPORTANCE_KEYWORDS
                   if kw in content_lower)
        # Normalise: 0 keywords = 0.0, 3+ = 1.0
        return min(1.0, hits / 3.0)

    @staticmethod
    def _role_score(message: Message) -> float:
        """Assistant answers are more important than user questions."""
        role_scores = {
            "assistant": 1.0,
            "user": 0.5,
            "system": 0.8,  # System prompt is important but static
        }
        return role_scores.get(message.role, 0.3)

    def _evict_lowest(self) -> None:
        """Remove the message with the lowest importance score."""
        lowest_idx = 0
        lowest_score = self._messages[0][1]
        for i, (_, score) in enumerate(self._messages):
            if score < lowest_score:
                lowest_score = score
                lowest_idx = i
        self._messages.pop(lowest_idx)


# ==============================================================================
# SECTION 7 -- MemoryManager (unified controller)
# ==============================================================================
#
# Design choice: The manager holds ONE strategy at a time and provides
# a single interface for the agent loop.  To change strategy you call
# switch_strategy() -- the manager handles migration automatically.

STRATEGY_MAP = {
    "full":        FullHistoryMemory,
    "sliding":     SlidingWindowMemory,
    "summarizing": SummarizingMemory,
    "importance":  ImportanceWeightedMemory,
}


class MemoryManager:
    """
    Unified memory controller.
    
    The agent loop talks to this manager, never directly to a strategy.
    This means you can swap strategies at runtime without changing
    any agent code.
    
    Usage:
        manager = MemoryManager(strategy="sliding", window_size=10)
        manager.add(Message(role="user", content="Hello"))
        context = manager.get_context()  # List[Message] for the LLM
        print(manager.stats())
    """

    def __init__(
        self,
        strategy: str = "sliding",
        **strategy_kwargs,
    ):
        """
        Args:
            strategy: One of "full", "sliding", "summarizing", "importance".
            **strategy_kwargs: Passed to the strategy constructor.
        """
        if strategy not in STRATEGY_MAP:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(STRATEGY_MAP.keys())}")
        
        self._strategy_name = strategy
        self._strategy = self._build_strategy(strategy, strategy_kwargs)
        self._all_messages: List[Message] = []  # Internal log of everything

    # ----- public interface ---------------------------------------------------

    def add(self, message: Message) -> None:
        """Add a message to memory (also logs it internally for migration)."""
        self._all_messages.append(message)
        self._strategy.add(message)

    def add_dict(self, role: str, content: str) -> None:
        """Convenience: add a message from a raw dict (backward compat)."""
        self.add(Message(role=role, content=content))

    def get_context(self) -> List[Message]:
        """Get the messages to send to the LLM."""
        return self._strategy.get_context()

    def get_context_dicts(self) -> List[Dict[str, str]]:
        """Get context as raw dicts (for OpenAI API compatibility)."""
        return [m.to_dict() for m in self.get_context()]

    def stats(self) -> Dict[str, Any]:
        """Get diagnostic info."""
        s = self._strategy.stats()
        s["total_messages_logged"] = len(self._all_messages)
        return s

    def switch_strategy(self, new_strategy: str, **new_kwargs) -> None:
        """
        Swap to a different strategy at runtime.
        
        The manager automatically migrates all logged messages to the
        new strategy so no history is lost during the switch.
        """
        if new_strategy not in STRATEGY_MAP:
            raise ValueError(f"Unknown strategy: {new_strategy}")
        
        old_stats = self.stats()
        self._strategy_name = new_strategy
        self._strategy = self._build_strategy(new_strategy, new_kwargs)
        
        # Migrate: replay all logged messages into the new strategy
        for msg in self._all_messages:
            self._strategy.add(msg)
        
        new_stats = self.stats()
        return {
            "old": old_stats,
            "new": new_stats,
            "migrated_messages": len(self._all_messages),
        }

    def clear(self) -> None:
        """Wipe all memory."""
        self._all_messages.clear()
        self._strategy = self._build_strategy(self._strategy_name, {})

    # ----- internal -----------------------------------------------------------

    @staticmethod
    def _build_strategy(name: str, kwargs: dict) -> MemoryStrategy:
        """Create a strategy instance by name."""
        cls = STRATEGY_MAP[name]
        return cls(**kwargs) if kwargs else cls()


# ==============================================================================
# SECTION 8 -- Backward compatibility (agent_loop.py still works)
# ==============================================================================
#
# These functions mirror the old API so agent_loop.py does not break.
# They are thin wrappers around SlidingWindowMemory.

# Module-level sliding window instance (shared across calls)
_default_memory: Optional[SlidingWindowMemory] = None


def _get_default_memory(window_size: int) -> SlidingWindowMemory:
    """Get or create the default sliding window memory."""
    global _default_memory
    if _default_memory is None or _default_memory._window_size != window_size:
        _default_memory = SlidingWindowMemory(window_size=window_size)
    return _default_memory


def build_prompt(
    messages: List[Dict[str, str]],
    window_size: int = 10,
) -> List[Dict[str, str]]:
    """
    Backward-compatible wrapper.
    
    Converts raw dicts to Messages, runs them through a sliding window,
    and converts back to dicts.  This is what agent_loop.py calls.
    """
    memory = _get_default_memory(window_size)
    # Clear and repopulate (the old messages list is the full history)
    memory._messages = []
    for msg in messages:
        memory.add(Message(role=msg["role"], content=msg["content"]))
    return [m.to_dict() for m in memory.get_context()]


def trim_for_api(
    messages: List[Dict[str, str]],
    window_size: int = 10,
    max_total_tokens: int = 60000,
) -> List[Dict[str, str]]:
    """
    Backward-compatible wrapper with token budget safety.
    """
    # Quick estimate
    estimated_tokens = sum(len(m.get("content", "")) for m in messages) // 4

    if estimated_tokens < max_total_tokens:
        return build_prompt(messages, window_size)

    # Over budget: use a smaller window
    return build_prompt(messages, window_size=max(3, window_size // 2))


# ==============================================================================
# SECTION 9 -- 30-turn stress test
# ==============================================================================
#
# Simulates a long conversation and measures how each strategy handles it.
# This is the validation that the notebook teaches -- we measure, not just build.

def _generate_fake_turn(step: int) -> Tuple[str, str, str]:
    """Generate a fake user question and assistant answer for testing."""
    questions = [
        f"What is the result of calculation step {step}?",
        f"Can you explain concept {step}?",
        f"Please analyze data point {step} in detail.",
        f"How does step {step} relate to the overall goal?",
        f"Summarize findings from step {step}.",
    ]
    answers = [
        f"The result of step {step} is {step * 42}. This was computed using the standard algorithm.",
        f"Concept {step} is important because it builds on the previous {step-1} concepts. Here is the explanation...",
        f"Data point {step} shows a clear trend. The values are increasing linearly with a slope of {step * 0.5}.",
        f"Step {step} relates to the goal by providing the {step}th piece of evidence needed for the conclusion.",
        f"Summary of step {step}: we found {step} key insights. The most important is that X equals Y.",
    ]
    q_idx = step % len(questions)
    return questions[q_idx], answers[q_idx], f"Observation from tool call {step}"


def stress_test(num_turns: int = 30) -> Dict[str, Any]:
    """
    Run a simulated conversation through all 4 strategies and compare.
    
    Args:
        num_turns: How many conversation turns to simulate (default 30).
    
    Returns:
        Dict with per-strategy stats and a comparison summary.
    """
    print(f"\n{'=' * 60}")
    print(f"MEMORY STRESS TEST -- {num_turns} turns")
    print(f"{'=' * 60}\n")

    results = {}

    for strategy_name in ["full", "sliding", "summarizing", "importance"]:
        manager = MemoryManager(strategy=strategy_name)
        
        turn_start = time.time()
        
        for i in range(1, num_turns + 1):
            q, a, obs = _generate_fake_turn(i)
            manager.add(Message(role="user", content=q))
            manager.add(Message(role="assistant", content=a))
            manager.add(Message(role="user", content=f"Tool returned: {obs}"))
        
        elapsed = time.time() - turn_start
        context = manager.get_context()
        stats = manager.stats()
        total_tokens = sum(m.estimate_tokens() for m in context)
        
        results[strategy_name] = {
            "context_messages": len(context),
            "total_tokens": total_tokens,
            "stats": stats,
            "elapsed_seconds": round(elapsed, 4),
        }
        
        print(f"  [{strategy_name:15s}] context={len(context):4d} msgs, "
              f"tokens={total_tokens:6d}, time={elapsed:.4f}s")

    # Comparison summary
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    
    # Find the winner for each metric
    min_tokens = min(r["total_tokens"] for r in results.values())
    min_context = min(r["context_messages"] for r in results.values())
    min_time = min(r["elapsed_seconds"] for r in results.values())
    
    for metric, best_val, label in [
        ("total_tokens", min_tokens, "Fewest tokens"),
        ("context_messages", min_context, "Smallest context"),
        ("elapsed_seconds", min_time, "Fastest"),
    ]:
        winners = [name for name, r in results.items() if r[metric] == best_val]
        print(f"  {label}: {', '.join(winners)} ({best_val})")
    
    print(f"\n  Full history tokens:    {results['full']['total_tokens']:>6d}")
    print(f"  Sliding window tokens:  {results['sliding']['total_tokens']:>6d}  "
          f"({'-' if results['sliding']['total_tokens'] < results['full']['total_tokens'] else '+'}"
          f"{results['sliding']['total_tokens'] - results['full']['total_tokens']:+d})")
    print(f"  Summarizing tokens:     {results['summarizing']['total_tokens']:>6d}  "
          f"({'-' if results['summarizing']['total_tokens'] < results['full']['total_tokens'] else '+'}"
          f"{results['summarizing']['total_tokens'] - results['full']['total_tokens']:+d})")
    print(f"  Importance tokens:      {results['importance']['total_tokens']:>6d}  "
          f"({'-' if results['importance']['total_tokens'] < results['full']['total_tokens'] else '+'}"
          f"{results['importance']['total_tokens'] - results['full']['total_tokens']:+d})")
    
    return results


# ==============================================================================
# SECTION 10 -- Quick demo (run this file directly)
# ==============================================================================

if __name__ == "__main__":
    # Quick sanity check: build a manager and add some messages
    print("Creating MemoryManager with 'sliding' strategy (window=5)")
    manager = MemoryManager(strategy="sliding", window_size=5)
    
    for i in range(8):
        manager.add(Message(role="user" if i % 2 == 0 else "assistant",
                           content=f"Message {i}: " + "x" * (20 + i * 10)))
    
    context = manager.get_context()
    print(f"  Added 8 messages, context has {len(context)} messages")
    print(f"  Stats: {manager.stats()}")
    
    # Switch to importance strategy
    print("\nSwitching to 'importance' strategy...")
    migration = manager.switch_strategy("importance")
    print(f"  Migrated {migration['migrated_messages']} messages")
    print(f"  New stats: {manager.stats()}")
    
    # Run the stress test
    stress_test(num_turns=30)
