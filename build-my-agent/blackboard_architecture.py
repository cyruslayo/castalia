import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple, Union
from collections import defaultdict
import json

# Reuse our config for LLM generation
try:
    from config import get_model, get_client
    client = get_client()
    MODEL_NAME = get_model()
except ImportError:
    # Fallback for isolated testing
    client = None
    MODEL_NAME = "hermes-model"

def generate(messages, max_new_tokens=512, temperature=0.7):
    """Generate a response from the model using the centralized config."""
    if not client:
        return "LLM Client not initialized. [MOCKED RESPONSE]"
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    # Handle thinking models where content might be in reasoning field
    content = response.choices[0].message.content
    if content is None and hasattr(response.choices[0].message, 'reasoning'):
        content = response.choices[0].message.reasoning
    return content

@dataclass
class BlackboardEntry:
    """A single entry on the blackboard with metadata."""
    key: str
    value: Any
    author: str
    timestamp: float = field(default_factory=time.time)
    version: int = 1

    def __repr__(self):
        preview = str(self.value)[:60]
        return f"Entry({self.key}={preview}..., by={self.author}, v{self.version})"

class Blackboard:
    """Shared knowledge store for multi-agent collaboration."""

    def __init__(self, name: str = "main"):
        self.name = name
        self._store: Dict[str, BlackboardEntry] = {}
        self._history: Dict[str, List[BlackboardEntry]] = defaultdict(list)
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._access_control: Dict[str, Dict[str, set]] = {}  # key -> {readers: set, writers: set}
        self._event_log: List[Dict] = []

    def set(self, key: str, value: Any, author: str) -> None:
        """Write a value to the blackboard."""
        # Check write access
        if key in self._access_control:
            writers = self._access_control[key].get("writers", set())
            if writers and author not in writers:
                self._event_log.append({
                    "event": "ACCESS_DENIED", "key": key, 
                    "author": author, "action": "write", "time": time.time()
                })
                print(f"  [ACCESS DENIED] {author} cannot write to '{key}'")
                return

        timestamp = time.time()
        
        # Track version
        version = 1
        if key in self._store:
            version = self._store[key].version + 1

        entry = BlackboardEntry(key, value, author, timestamp, version)

        # Store and log history
        self._store[key] = entry
        self._history[key].append(entry)
        self._event_log.append({
            "event": "WRITE", "key": key, "author": author,
            "version": version, "timestamp": timestamp
        })

        # Notify subscribers
        for callback in self._subscribers.get(key, []):
            callback(key, value, author)

    def get(self, key: str, reader: str = "system") -> Optional[Any]:
        """Read a value from the blackboard."""
        if key not in self._store:
            return None

        # Check read access
        if key in self._access_control:
            readers = self._access_control[key].get("readers", set())
            if readers and reader not in readers:
                self._event_log.append({
                    "event": "ACCESS_DENIED", "key": key, 
                    "author": reader, "action": "read", "time": time.time()
                })
                return None

        self._event_log.append({
            "event": "READ", "key": key, "reader": reader, "time": time.time()
        })
        return self._store[key].value

    def subscribe(self, key: str, callback: Callable):
        """Subscribe to changes on a key."""
        self._subscribers[key].append(callback)

    def set_access(self, key: str, readers: set = None, writers: set = None):
        """Set access control for a key."""
        self._access_control[key] = {
            "readers": readers or set(),
            "writers": writers or set()
        }

    def get_history(self, key: str) -> List[Dict]:
        """Get version history for a key."""
        return [{"version": e.version, "value": str(e.value)[:80],
                 "author": e.author, "time": e.timestamp}
                for e in self._history.get(key, [])]

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def snapshot(self) -> Dict[str, str]:
        """Get a snapshot of all current values."""
        return {k: str(v.value)[:100] for k, v in self._store.items()}

    def stats(self) -> Dict[str, Any]:
        return {
            "total_keys": len(self._store),
            "total_writes": sum(len(h) for h in self._history.values()),
            "total_events": len(self._event_log),
            "authors": list(set(e.author for e in self._store.values()))
        }

class BlackboardAgent:
    """Agent that reads context from blackboard, processes, and writes results back."""

    def __init__(self, name: str, role: str, read_keys: List[str], write_key: str):
        self.name = name
        self.role = role
        self.read_keys = read_keys
        self.write_key = write_key
        self.runs = 0

    def run(self, blackboard: Blackboard, task: str = "") -> str:
        """Read from blackboard, process, write results back."""
        self.runs += 1

        # Read context from blackboard
        context_parts = []
        for key in self.read_keys:
            value = blackboard.get(key, reader=self.name)
            if value is not None:
                context_parts.append(f"[{key}]: {value}")

        context = "\n".join(context_parts) if context_parts else "No prior context available."

        # Build prompt
        messages = [
            {"role": "system", "content": f"You are {self.name}, a {self.role}. Use the provided context to inform your work. Be concise (3-4 sentences)."},
            {"role": "user", "content": f"Context from shared knowledge:\n{context}\n\nTask: {task}"}
        ]

        t0 = time.time()
        response = generate(messages, max_new_tokens=250)
        elapsed = time.time() - t0

        # Write results to blackboard
        blackboard.set(self.write_key, response, self.name)

        print(f"  [{self.name}] Read: {self.read_keys} → Write: '{self.write_key}' ({elapsed:.1f}s)")
        return response

    def __repr__(self):
        return f"BBAgent({self.name}, reads={self.read_keys}, writes='{self.write_key}')"

class EventDrivenBlackboard(Blackboard):
    """Blackboard with event-driven agent activation."""

    def __init__(self, name: str = "event_driven"):
        super().__init__(name)
        self._agent_triggers: Dict[str, List[Tuple[BlackboardAgent, str]]] = defaultdict(list)
        self._activation_log: List[Dict] = []

    def register_trigger(self, watch_key: str, agent: BlackboardAgent, task: str):
        """Register an agent to activate when a key changes."""
        self._agent_triggers[watch_key].append((agent, task))

    def set(self, key: str, value: Any, author: str) -> None:
        """Write value and trigger registered agents."""
        super().set(key, value, author)

        # Trigger agents watching this key
        if key in self._agent_triggers:
            for agent, task in self._agent_triggers[key]:
                if agent.name != author:  # prevent self-triggering
                    print(f"    [TRIGGERED] {agent.name} (watching '{key}')")
                    self._activation_log.append({
                        "trigger_key": key, "agent": agent.name,
                        "caused_by": author, "time": time.time()
                    })
                    agent.run(self, task)

class ConflictResolvingBlackboard(Blackboard):
    """Blackboard with conflict resolution strategies."""

    STRATEGIES = ["last_write_wins", "keep_all", "llm_merge"]

    def __init__(self, name: str, strategy: str = "last_write_wins"):
        super().__init__(name)
        assert strategy in self.STRATEGIES
        self.strategy = strategy
        self.conflicts: List[Dict] = []

    def set(self, key: str, value: Any, author: str) -> None:
        if key in self._store and self._store[key].author != author:
            # Conflict detected
            existing = self._store[key]
            self.conflicts.append({
                "key": key,
                "existing_author": existing.author,
                "existing_value": str(existing.value)[:100],
                "new_author": author,
                "new_value": str(value)[:100],
                "strategy": self.strategy,
                "time": time.time()
            })
            print(f"  [CONFLICT] on '{key}': {existing.author} vs {author}")

            if self.strategy == "last_write_wins":
                super().set(key, value, author)

            elif self.strategy == "keep_all":
                # Store as list of dicts
                if isinstance(existing.value, list) and len(existing.value) > 0 and isinstance(existing.value[0], dict) and "author" in existing.value[0]:
                    merged = existing.value + [{"author": author, "value": value}]
                else:
                    merged = [
                        {"author": existing.author, "value": existing.value},
                        {"author": author, "value": value}
                    ]
                super().set(key, merged, "system_merge")

            elif self.strategy == "llm_merge":
                merged = self._llm_resolve(key, existing.value, existing.author, value, author)
                super().set(key, merged, "llm_merge")
        else:
            super().set(key, value, author)

    def _llm_resolve(self, key: str, val1: Any, author1: str, val2: Any, author2: str) -> str:
        messages = [
            {"role": "system", "content": "You reconcile conflicting information. Combine both perspectives into a coherent merged version. Be concise."},
            {"role": "user", "content": f"Key: {key}\n[{author1}]: {val1}\n[{author2}]: {val2}\n\nMerge these into one coherent entry."}
        ]
        return generate(messages, max_new_tokens=200)
