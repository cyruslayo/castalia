"""
Runtime contracts for the integrated Castalia agent system.

These dataclasses are the shared boundary between the older notebook-specific
agents and the new runtime/orchestration layer.  They intentionally stay small
and dependency-light so every module can import them safely.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def new_id(prefix: str) -> str:
    """Return a compact unique id with a readable prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@dataclass
class AgentRequest:
    """A normalized request to any agent or runtime."""

    task: str
    user_id: str = "default"
    session_id: str = "default"
    strategy: str = "auto"
    max_steps: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: new_id("req"))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallRecord:
    """Canonical record of a tool call emitted by the runtime."""

    tool: str
    params: Dict[str, Any]
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    call_id: str = field(default_factory=lambda: new_id("tool"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepRecord:
    """Canonical trace step for agent reasoning/action."""

    step: int
    action: str
    content: Any = None
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_id: str = field(default_factory=lambda: new_id("step"))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    """One return shape for all agents, tools, and orchestration patterns."""

    answer: str
    success: bool
    strategy_used: str
    steps: List[Any] = field(default_factory=list)
    tool_calls: List[Any] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    request_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def finish(self) -> "AgentResult":
        if self.finished_at is None:
            self.finished_at = time.time()
        self.metadata.setdefault("duration_seconds", self.finished_at - self.started_at)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentContext:
    """Runtime context passed to adapter agents."""

    request: AgentRequest
    memory_context: List[Dict[str, str]] = field(default_factory=list)
    tool_runtime: Any = None
    memory_hub: Any = None
    event_bus: Any = None
    optimizer: Any = None
    blackboard: Any = None
    runtime_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeEvent:
    """Structured event emitted by the integrated runtime."""

    event_type: str
    actor: str
    payload: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: new_id("evt"))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
