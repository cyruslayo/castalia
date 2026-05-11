"""Unified memory facade: short-term + optional long-term + optional graph memory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from memory import MemoryManager


class MemoryHub:
    """Facade over the existing memory notebooks.

    The hub keeps the short-term MemoryManager always enabled and lazily enables
    LongTermMemory/HybridMemory if their dependencies are available.
    """

    def __init__(
        self,
        short_strategy: str = "sliding",
        persist_path: Optional[str] = None,
        enable_long_term: bool = True,
        enable_graph: bool = False,
        **short_kwargs,
    ):
        self.short = MemoryManager(strategy=short_strategy, **short_kwargs)
        self.long_term = None
        self.graph = None
        self.persist_path = persist_path

        if enable_long_term:
            try:
                from long_term_memory import LongTermMemory
                self.long_term = LongTermMemory(persist_path=persist_path) if persist_path else LongTermMemory()
            except Exception as e:
                self.long_term_error = f"LongTermMemory unavailable: {type(e).__name__}: {e}"
            else:
                self.long_term_error = None

        if enable_graph:
            try:
                from graph_memory import HybridMemory
                self.graph = HybridMemory()
            except Exception as e:
                self.graph_error = f"Graph/HybridMemory unavailable: {type(e).__name__}: {e}"
            else:
                self.graph_error = None

    def add_message(self, role: str, content: str) -> None:
        self.short.add_dict(role, content)

    def short_context(self) -> List[Dict[str, str]]:
        return self.short.get_context_dicts()

    def recall(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """Return compact memory snippets as chat messages for prompt injection."""
        snippets = []

        if self.long_term is not None:
            try:
                recalled = self.long_term.recall(query, k=k)
                if recalled:
                    snippets.append("Long-term memory:\n" + "\n".join(str(x)[:500] for x in recalled))
            except Exception as e:
                snippets.append(f"Long-term memory recall failed: {type(e).__name__}: {e}")

        if self.graph is not None:
            try:
                recalled = self.graph.query(query, k=k)
                if recalled:
                    snippets.append("Graph/hybrid memory:\n" + "\n".join(str(x)[:500] for x in recalled))
            except Exception as e:
                snippets.append(f"Graph memory recall failed: {type(e).__name__}: {e}")

        if not snippets:
            return []
        return [{"role": "system", "content": "\n\n".join(snippets)}]

    def store_result(self, request, result) -> None:
        text = f"Task: {request.task}\nAnswer: {result.answer}\nSuccess: {result.success}"
        if self.long_term is not None:
            try:
                # LongTermMemory API supports store_episode in current implementation.
                if hasattr(self.long_term, "store_episode"):
                    self.long_term.store_episode(text, importance=1.0)
                if hasattr(self.long_term, "save"):
                    self.long_term.save()
            except Exception:
                pass
        if self.graph is not None:
            try:
                if hasattr(self.graph, "store"):
                    self.graph.store(text)
            except Exception:
                pass

    def stats(self) -> Dict[str, Any]:
        stats = {"short": self.short.stats()}
        if self.long_term is not None and hasattr(self.long_term, "stats"):
            try:
                stats["long_term"] = self.long_term.stats()
            except Exception as e:
                stats["long_term_error"] = str(e)
        if self.graph is not None:
            try:
                stats["graph"] = str(self.graph)
            except Exception as e:
                stats["graph_error"] = str(e)
        return stats
