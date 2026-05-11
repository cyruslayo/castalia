"""Runtime wrapper for the shared blackboard notebook implementation."""

from __future__ import annotations

from typing import Any, Optional

from blackboard_architecture import ConflictResolvingBlackboard


class BlackboardRuntime:
    """Small facade used by AgentRuntime/capstone orchestration."""

    def __init__(self, name: str = "runtime_blackboard", strategy: str = "last_write_wins"):
        self.board = ConflictResolvingBlackboard(name, strategy=strategy)

    def publish(self, key: str, value: Any, author: str = "runtime") -> dict:
        self.board.set(key, value, author=author)
        return {"success": True, "key": key, "author": author}

    def read(self, key: str, default: Any = None) -> Any:
        value = self.board.get(key)
        return default if value is None else value

    def snapshot(self) -> dict:
        return self.board.snapshot()

    def keys(self) -> list:
        return list(self.snapshot().keys())
