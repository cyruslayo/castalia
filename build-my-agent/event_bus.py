"""Simple in-process EventBus for runtime observability.

Notebook 31 will likely expand this into lifecycle management and JSON trace
shipping.  For now it provides the minimum contract needed by the integrated
runtime and tests.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional

from runtime_contracts import RuntimeEvent


class EventBus:
    """Small pub/sub event bus with in-memory event history."""

    def __init__(self, max_events: int = 10_000):
        self.max_events = max_events
        self.events: List[RuntimeEvent] = []
        self._subscribers: Dict[str, List[Callable[[RuntimeEvent], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[RuntimeEvent], None]) -> None:
        self._subscribers[event_type].append(handler)

    def emit(self, event_type: str, actor: str, payload: Optional[dict] = None,
             request_id: Optional[str] = None) -> RuntimeEvent:
        event = RuntimeEvent(
            event_type=event_type,
            actor=actor,
            payload=payload or {},
            request_id=request_id,
        )
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        for handler in self._subscribers.get(event_type, []):
            handler(event)
        for handler in self._subscribers.get("*", []):
            handler(event)
        return event

    def recent(self, limit: int = 50, event_type: Optional[str] = None,
               request_id: Optional[str] = None) -> List[dict]:
        events = self.events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if request_id:
            events = [e for e in events if e.request_id == request_id]
        return [e.to_dict() for e in events[-limit:]]

    def stats(self) -> dict:
        counts = defaultdict(int)
        for event in self.events:
            counts[event.event_type] += 1
        return {"total_events": len(self.events), "by_type": dict(counts)}
