"""Notebook 31 runtime infrastructure: registry, logger, lifecycle manager."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from event_bus import EventBus


@dataclass
class LogEntry:
    actor: str
    level: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class AgentLogger:
    """Structured logger with per-actor traces and JSON export."""

    def __init__(self, min_level: str = "DEBUG"):
        self.min_level = min_level.upper()
        self._priority = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        self._entries: List[LogEntry] = []
        self._traces: Dict[str, List[LogEntry]] = defaultdict(list)

    def log(self, actor: str, level: str, message: str, metadata: Optional[dict] = None) -> LogEntry:
        level = level.upper()
        if self._priority.get(level, 0) < self._priority.get(self.min_level, 10):
            return LogEntry(actor=actor, level=level, message=message, metadata=metadata or {})
        entry = LogEntry(actor=actor, level=level, message=message, metadata=metadata or {})
        self._entries.append(entry)
        self._traces[actor].append(entry)
        return entry

    def trace(self, actor: str) -> List[dict]:
        return [entry.to_dict() for entry in self._traces.get(actor, [])]

    def errors(self) -> List[dict]:
        return [entry.to_dict() for entry in self._entries if self._priority.get(entry.level, 0) >= 40]

    def export_json(self) -> str:
        return json.dumps({actor: self.trace(actor) for actor in self._traces}, indent=2)

    def stats(self) -> dict:
        counts = defaultdict(int)
        for entry in self._entries:
            counts[entry.level] += 1
        return {"total_entries": len(self._entries), "by_level": dict(counts), "actors": sorted(self._traces.keys())}


@dataclass
class AgentRegistration:
    name: str
    strategy: str
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable[..., Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)


class AgentRegistry:
    """Registry of runtime-managed logical agents."""

    def __init__(self):
        self._agents: Dict[str, AgentRegistration] = {}

    def register(
        self,
        name: str,
        strategy: str,
        description: str = "",
        capabilities: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable[..., Any]] = None,
    ) -> AgentRegistration:
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered")
        registration = AgentRegistration(
            name=name,
            strategy=strategy,
            description=description,
            capabilities=list(capabilities or []),
            config=dict(config or {}),
            handler=handler,
        )
        self._agents[name] = registration
        return registration

    def get(self, name: str) -> AgentRegistration:
        if name not in self._agents:
            raise KeyError(f"Unknown agent: {name}")
        return self._agents[name]

    def list_agents(self) -> List[dict]:
        return [registration.to_dict() for registration in self._agents.values()]

    def find_by_capability(self, capability: str) -> List[str]:
        return [name for name, registration in self._agents.items() if capability in registration.capabilities]

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)


@dataclass
class LifecycleRecord:
    name: str
    strategy: str
    status: str = "registered"
    started_at: float = 0.0
    restart_count: int = 0
    last_error: Optional[str] = None
    last_health: dict = field(default_factory=dict)
    tasks_completed: int = 0
    errors: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class AgentLifecycleManager:
    """Start/stop/restart/health lifecycle for runtime-registered agents."""

    def __init__(self, registry: AgentRegistry, event_bus: Optional[EventBus] = None, logger: Optional[AgentLogger] = None):
        self.registry = registry
        self.event_bus = event_bus or EventBus()
        self.logger = logger or AgentLogger()
        self.records: Dict[str, LifecycleRecord] = {}

    def register(self, name: str) -> LifecycleRecord:
        registration = self.registry.get(name)
        record = LifecycleRecord(name=name, strategy=registration.strategy)
        self.records[name] = record
        self.logger.log(name, "INFO", "Agent registered", {"strategy": registration.strategy})
        self.event_bus.emit("agent.registered", name, {"strategy": registration.strategy})
        return record

    def ensure_registered(self, name: str) -> LifecycleRecord:
        if name not in self.records:
            return self.register(name)
        return self.records[name]

    def start(self, name: str) -> dict:
        record = self.ensure_registered(name)
        record.status = "running"
        record.started_at = time.time()
        record.last_error = None
        self.logger.log(name, "INFO", "Agent started", {"strategy": record.strategy})
        self.event_bus.emit("agent.started", name, {"strategy": record.strategy})
        return record.to_dict()

    def stop(self, name: str) -> dict:
        record = self.ensure_registered(name)
        record.status = "stopped"
        self.logger.log(name, "INFO", "Agent stopped")
        self.event_bus.emit("agent.stopped", name, {"strategy": record.strategy})
        return record.to_dict()

    def restart(self, name: str) -> dict:
        record = self.ensure_registered(name)
        record.status = "restarting"
        record.restart_count += 1
        self.logger.log(name, "WARNING", "Agent restarting", {"restart_count": record.restart_count})
        self.event_bus.emit("agent.restarting", name, {"restart_count": record.restart_count})
        return self.start(name)

    def mark_task_completed(self, name: str) -> None:
        record = self.ensure_registered(name)
        record.tasks_completed += 1

    def mark_error(self, name: str, error: str) -> None:
        record = self.ensure_registered(name)
        record.errors += 1
        record.last_error = error
        record.status = "failed"
        self.logger.log(name, "ERROR", "Agent failed", {"error": error})
        self.event_bus.emit("agent.failed", name, {"error": error, "strategy": record.strategy})

    def health(self, name: str) -> dict:
        record = self.ensure_registered(name)
        health = {
            "name": name,
            "strategy": record.strategy,
            "status": record.status,
            "healthy": record.status == "running",
            "uptime_seconds": max(0.0, time.time() - record.started_at) if record.started_at else 0.0,
            "restart_count": record.restart_count,
            "tasks_completed": record.tasks_completed,
            "errors": record.errors,
            "last_error": record.last_error,
        }
        record.last_health = health
        self.event_bus.emit("agent.health_checked", name, health)
        return health

    def list_agents(self) -> List[dict]:
        return [record.to_dict() for record in self.records.values()]
