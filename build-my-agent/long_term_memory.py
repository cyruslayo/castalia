"""
Long-Term Memory for Agents (Notebook 11)

Two types of memory that persist beyond the context window:
  1. EpisodicMemory  -- timestamped experiences stored as embeddings
  2. SemanticMemory  -- deduplicated facts in a searchable store

Plus a unified LongTermMemory manager that:
  - Combines both stores
  - Saves/loads to disk (cross-session persistence)
  - Applies importance decay (memories fade unless reinforced)
  - Recalls by semantic similarity (not just keywords)

Dependencies: faiss-cpu, sklearn (TF-IDF for embeddings)
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from fastembed.embedding import TextEmbedding


# ==============================================================================
# SECTION 1 -- Embedding Engine (BGE via fastembed)
# ==============================================================================
#
# We use BAAI/bge-small-en-v1.5 via fastembed (ONNX runtime, no PyTorch):
#   - 384 dimensions, runs on CPU
#   - Downloads ~130MB model on first use (~70s)
#   - Real semantic embeddings: "king - man + woman ~= queen"
#   - Unlike TF-IDF, captures synonymy and paraphrase
#
# The interface is the same (fit/encode/encode_batch/dimension) so swapping
# TF-IDF for BGE only changes this one class.

class BGEEmbedder:
    """BAAI/bge-small-en-v1.5 embedder via fastembed (ONNX, CPU-only)."""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DIMENSION = 384  # bge-small-en-v1.5 produces 384-dim vectors

    def __init__(self):
        self._model: Optional[TextEmbedding] = None

    def _load_model(self) -> TextEmbedding:
        """Lazy-load the ONNX model (first call downloads weights)."""
        if self._model is None:
            self._model = TextEmbedding(model_name=self.MODEL_NAME)
        return self._model

    def fit(self, texts: List[str]) -> None:
        """No-op -- BGE is pre-trained and requires no corpus fitting."""
        pass  # kept for API compatibility (TF-IDF required fitting, BGE does not)

    def encode(self, text: str) -> np.ndarray:
        """
        Convert a single text into a 384-dim embedding vector.
        Returns a numpy array of shape (384,).
        """
        model = self._load_model()
        # model.embed(texts) returns an iterator/generator of arrays
        vec = list(model.embed([text]))[0]
        return np.asarray(vec, dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts into a 2D array of shape (n_texts, 384).
        """
        if not texts:
            return np.zeros((0, self.DIMENSION), dtype=np.float32)
        model = self._load_model()
        vecs = list(model.embed(texts))
        # Stack into 2D array: (n_texts, 384)
        return np.stack(vecs).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self.DIMENSION


# ==============================================================================
# SECTION 2 -- Episodic Memory (FAISS-backed experience store)
# ==============================================================================
#
# An "episode" is a single notable experience: what happened, when, and how
# important it was. Episodes are stored as TF-IDF vectors in a FAISS index
# so we can search them by semantic similarity.
#
# Why FAISS?
#   - FAISS (Facebook AI Similarity Search) indexes vectors for fast retrieval.
#   - Without FAISS you'd compare every query to every episode (O(n) per query).
#   - With FAISS, search is O(log n) or better depending on the index type.
#   - We use IndexFlatIP (inner product) for exact search -- perfect for <10K
#     episodes. For millions of episodes you'd use IVF or HNSW indexes.

@dataclass
class Episode:
    """A single memorable experience."""
    text: str                       # What happened (human-readable)
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0        # 0.0 (trivial) to 1.0 (critical)
    _embedding: Optional[List[float]] = field(default=None, repr=False)

    @property
    def embedding(self) -> np.ndarray:
        """Lazily convert stored list back to numpy array."""
        if self._embedding is None:
            return np.array([], dtype=np.float32)
        return np.array(self._embedding, dtype=np.float32)

    @embedding.setter
    def embedding(self, vec: np.ndarray):
        self._embedding = vec.tolist()

    def age_days(self) -> float:
        """How many days old is this episode?"""
        return (time.time() - self.timestamp) / 86400.0

    def decayed_importance(self, half_life_days: float = 7.0) -> float:
        """
        Importance decays exponentially over time.
        After `half_life_days`, importance is halved.
        Formula: importance * 2^(-age / half_life)
        """
        age = self.age_days()
        return self.importance * math.pow(0.5, age / half_life_days)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "_embedding": self._embedding
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Episode':
        return cls(
            text=d["text"],
            timestamp=d["timestamp"],
            importance=d["importance"],
            _embedding=d.get("_embedding")
        )


class EpisodicMemory:
    """
    Stores experiences as searchable vectors.

    Design choice: we rebuild the FAISS index when new episodes are added.
    For a learning system this is simpler than incremental adds. In production
    you'd use faiss.IndexIDMap to add/remove episodes without rebuilding.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.episodes: List[Episode] = []
        # IndexFlatIP = exact search via inner product (cosine similarity when normalized)
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, text: str, embedding: np.ndarray, importance: float = 1.0) -> Episode:
        """
        Store a new episode.
        Returns the Episode object for reference.
        """
        episode = Episode(text=text, importance=importance, _embedding=embedding.tolist())
        self.episodes.append(episode)
        return episode

    def rebuild_index(self, embedder: BGEEmbedder) -> None:
        """
        Rebuild the FAISS index from all episodes.
        Call this after adding/removing episodes, or when embedder vocabulary changes.
        
        Design choice: we fit the embedder on ALL episode texts, then re-encode
        everything. TF-IDF is corpus-dependent -- the same word has different
        weights depending on what other documents exist. This ensures all
        embeddings share the same vocabulary space.
        """
        if not self.episodes:
            self.index = faiss.IndexFlatIP(embedder.dimension)
            return

        # Collect all texts and fit embedder on the FULL corpus
        texts = [ep.text for ep in self.episodes]
        embedder.fit(texts)
        
        # Encode ALL episodes with the fitted vocabulary
        vectors = embedder.encode_batch(texts)
        
        # Update each episode's stored embedding
        for ep, vec in zip(self.episodes, vectors):
            ep.embedding = vec
        
        # Create index with the ACTUAL vocabulary size
        self.index = faiss.IndexFlatIP(embedder.dimension)
        self.index.add(vectors)

    def recall(self, query: str, embedder: BGEEmbedder, k: int = 3) -> List[Tuple[Episode, float]]:
        """
        Search for the k most semantically similar episodes.
        Returns list of (Episode, similarity_score) tuples, sorted by score descending.
        """
        if not self.episodes or self.index.ntotal == 0:
            return []

        # Embed the query
        query_vec = embedder.encode(query)
        query_vec = np.array([query_vec], dtype=np.float32)

        # Normalize for cosine-like search (FAISS IndexFlatIP does inner product)
        faiss.normalize_L2(query_vec)

        # Search -- returns (distances, indices)
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.episodes):
                continue
            results.append((self.episodes[idx], float(distances[0][i])))

        return results

    def prune_by_importance(self, threshold: float = 0.1, half_life_days: float = 7.0) -> int:
        """
        Remove episodes whose decayed importance falls below threshold.
        Returns the number of episodes removed.
        """
        original_count = len(self.episodes)
        self.episodes = [
            ep for ep in self.episodes
            if ep.decayed_importance(half_life_days) >= threshold
        ]
        return original_count - len(self.episodes)

    def stats(self) -> Dict[str, Any]:
        """Diagnostic info about the episodic store."""
        if not self.episodes:
            return {"total_episodes": 0, "avg_importance": 0.0, "oldest_days": 0.0, "faiss_vectors": 0}
        importances = [ep.importance for ep in self.episodes]
        ages = [ep.age_days() for ep in self.episodes]
        return {
            "total_episodes": len(self.episodes),
            "avg_importance": round(sum(importances) / len(importances), 3),
            "oldest_days": round(max(ages), 2),
            "newest_days": round(min(ages), 2),
            "faiss_vectors": self.index.ntotal
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"episodes": [ep.to_dict() for ep in self.episodes]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any], dimension: int = 384) -> 'EpisodicMemory':
        memory = cls(dimension=dimension)
        memory.episodes = [Episode.from_dict(ep) for ep in d.get("episodes", [])]
        return memory


# ==============================================================================
# SECTION 3 -- Semantic Memory (Deduplicated fact store)
# ==============================================================================
#
# Semantic memory stores facts, not experiences. Key differences:
#   - Episodic: "User asked me to calculate 987*654 on May 6" (specific event)
#   - Semantic: "987 * 654 = 645498" (general knowledge)
#
# Facts are deduplicated: storing the same fact twice increments a counter
# and updates the timestamp (reinforcement). This implements a simple
# "spaced repetition" mechanism -- frequently recalled facts persist longer.

@dataclass
class Fact:
    """A single piece of knowledge."""
    key: str                        # Unique identifier (e.g., "speed_of_light")
    value: str                      # The fact content
    source: str = "unknown"         # Where it came from (e.g., "user", "web", "derived")
    timestamps: List[float] = field(default_factory=list)  # When it was stored/reinforced
    confidence: float = 1.0         # 0.0 (doubtful) to 1.0 (verified)

    @property
    def reinforcement_count(self) -> int:
        return len(self.timestamps)

    @property
    def last_updated(self) -> float:
        return self.timestamps[-1] if self.timestamps else 0.0

    def reinforce(self) -> None:
        """Record a new timestamp (fact was recalled or verified)."""
        self.timestamps.append(time.time())
        # Slight confidence boost on reinforcement (cap at 1.0)
        self.confidence = min(1.0, self.confidence + 0.05)

    def age_days(self) -> float:
        return (time.time() - self.last_updated) / 86400.0 if self.last_updated else 999.0

    def decayed_confidence(self, half_life_days: float = 30.0) -> float:
        """Confidence decays over time unless reinforced."""
        return self.confidence * math.pow(0.5, self.age_days() / half_life_days)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "timestamps": self.timestamps,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Fact':
        return cls(
            key=d["key"],
            value=d["value"],
            source=d.get("source", "unknown"),
            timestamps=d.get("timestamps", []),
            confidence=d.get("confidence", 1.0)
        )


class SemanticMemory:
    """
    Deduplicated fact store with semantic search capability.
    Facts are stored by key (fast lookup) AND embedded for similarity search.
    """

    def __init__(self, dimension: int = 384):
        self.facts: Dict[str, Fact] = {}  # key -> Fact
        self.dimension = dimension
        self._embedder = BGEEmbedder()
        self._index = faiss.IndexFlatIP(dimension)
        self._fact_vectors: List[np.ndarray] = []  # parallel to facts list

    def store_fact(self, key: str, value: str, source: str = "unknown") -> Fact:
        """
        Store or reinforce a fact. If the key already exists, we update the
        timestamps and boost confidence rather than creating a duplicate.
        """
        if key in self.facts:
            fact = self.facts[key]
            fact.reinforce()
            fact.value = value  # Update in case the value changed
            return fact

        fact = Fact(key=key, value=value, source=source)
        fact.reinforce()  # Timestamp the creation so decay works correctly
        self.facts[key] = fact
        return fact

    def get_fact(self, key: str) -> Optional[Fact]:
        """Exact lookup by key. O(1)."""
        return self.facts.get(key)

    def forget_fact(self, key: str) -> bool:
        """Remove a fact. Returns True if it existed, False otherwise."""
        if key in self.facts:
            del self.facts[key]
            return True
        return False

    def recall_by_query(self, query: str, k: int = 3) -> List[Tuple[Fact, float]]:
        """
        Find the k most semantically similar facts to the query.
        Returns list of (Fact, similarity_score) tuples.
        """
        if not self.facts:
            return []

        # Rebuild embeddings if needed
        self._rebuild_index()

        # Embed query
        query_vec = self._embedder.encode(query)
        query_vec = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # Search
        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(query_vec, k)

        fact_list = list(self.facts.values())
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(fact_list):
                continue
            results.append((fact_list[idx], float(distances[0][i])))

        return results

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from current facts."""
        facts_list = list(self.facts.values())
        if not facts_list:
            self._index = faiss.IndexFlatIP(self.dimension)
            return

        texts = [f"{f.key}: {f.value}" for f in facts_list]
        self._embedder.fit(texts)
        
        # Create index with the embedder's ACTUAL vocabulary size
        self._index = faiss.IndexFlatIP(self._embedder.dimension)
        vectors = self._embedder.encode_batch(texts)

        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        self._index.add(vectors)

    def prune_by_confidence(self, threshold: float = 0.1, half_life_days: float = 30.0) -> int:
        """Remove facts whose decayed confidence falls below threshold."""
        original_count = len(self.facts)
        to_remove = [
            key for key, fact in self.facts.items()
            if fact.decayed_confidence(half_life_days) < threshold
        ]
        for key in to_remove:
            del self.facts[key]
        return original_count - len(self.facts)

    def consolidate(self) -> int:
        """
        Merge redundant facts. Two facts are redundant if they have very
        similar keys or values. Returns number of merges performed.
        """
        if len(self.facts) < 2:
            return 0

        merges = 0
        keys = list(self.facts.keys())
        for i, k1 in enumerate(keys):
            if k1 not in self.facts:
                continue
            for k2 in keys[i+1:]:
                if k2 not in self.facts:
                    continue
                # Simple redundancy check: similar keys or one value in another
                f1, f2 = self.facts[k1], self.facts[k2]
                if (k1.lower().replace("_", " ") == k2.lower().replace("_", " ") or
                    f1.value.lower() == f2.value.lower()):
                    # Merge: keep the one with higher confidence + more timestamps
                    if f2.reinforcement_count > f1.reinforcement_count:
                        self.facts[k1] = f2
                    else:
                        f1.reinforce()  # Boost the survivor
                    del self.facts[k2]
                    merges += 1
        return merges

    def stats(self) -> Dict[str, Any]:
        """Diagnostic info about the semantic store."""
        if not self.facts:
            return {"total_facts": 0, "avg_confidence": 0.0, "total_reinforcements": 0}
        confidences = [f.confidence for f in self.facts.values()]
        reinforcements = [f.reinforcement_count for f in self.facts.values()]
        return {
            "total_facts": len(self.facts),
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
            "total_reinforcements": sum(reinforcements),
            "most_reinforced": max(self.facts.values(), key=lambda f: f.reinforcement_count).key
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"facts": {k: v.to_dict() for k, v in self.facts.items()}}

    @classmethod
    def from_dict(cls, d: Dict[str, Any], dimension: int = 384) -> 'SemanticMemory':
        memory = cls(dimension=dimension)
        facts_data = d.get("facts", {})
        for key, fact_dict in facts_data.items():
            memory.facts[key] = Fact.from_dict(fact_dict)
        return memory


# ==============================================================================
# SECTION 4 -- Unified Long-Term Memory Manager
# ==============================================================================
#
# This is the single entry point for the agent. It wraps episodic + semantic
# memory and provides:
#   - store_episode(text, importance)  -- record an experience
#   - store_fact(key, value, source)   -- record knowledge
#   - recall(query, mode, k)           -- search both stores
#   - save(path) / load(path)          -- cross-session persistence
#   - decay()                          -- apply importance/confidence decay
#   - consolidate()                    -- merge redundant facts

class LongTermMemory:
    """
    Unified long-term memory manager combining episodic and semantic stores.

    Design choice: this is NOT a subclass of MemoryStrategy from memory.py.
    Short-term and long-term memory solve different problems:
      - Short-term: "What messages should I send to the LLM right now?"
      - Long-term: "What do I know across all sessions?"
    They are complementary, not interchangeable. An agent uses both.
    """

    def __init__(self, dimension: int = 384, persist_path: Optional[str] = None):
        self.episodic = EpisodicMemory(dimension=dimension)
        self.semantic = SemanticMemory(dimension=dimension)
        self.embedder = BGEEmbedder()
        self._persist_path = persist_path

    # --- Storing ---

    def store_episode(self, text: str, importance: float = 1.0) -> Episode:
        """Record a new experience."""
        embedding = self.embedder.encode(text)
        episode = self.episodic.add(text=text, embedding=embedding, importance=importance)
        # Rebuild the FAISS index so recall works
        self.episodic.rebuild_index(self.embedder)
        return episode

    def store_fact(self, key: str, value: str, source: str = "unknown") -> Fact:
        """Record a new piece of knowledge (or reinforce existing)."""
        return self.semantic.store_fact(key=key, value=value, source=source)

    def store_experiences_batch(self, texts: List[str]) -> None:
        """Add multiple episodes at once (more efficient than one-by-one)."""
        for text in texts:
            embedding = self.embedder.encode(text)
            self.episodic.add(text=text, embedding=embedding)
        # Rebuild once after all additions
        self.episodic.rebuild_index(self.embedder)

    # --- Recalling ---

    def recall_episodes(self, query: str, k: int = 3) -> List[Tuple[Episode, float]]:
        """Find the k most similar past experiences."""
        return self.episodic.recall(query=query, embedder=self.embedder, k=k)

    def recall_facts(self, query: str, k: int = 3) -> List[Tuple[Fact, float]]:
        """Find the k most similar facts."""
        return self.semantic.recall_by_query(query=query, k=k)

    def recall(self, query: str, mode: str = "both", k: int = 3) -> Dict[str, Any]:
        """
        Unified recall interface.
        mode: "episodes" | "facts" | "both"
        Returns a dict with the results.
        """
        result = {"query": query, "mode": mode}
        if mode in ("episodes", "both"):
            result["episodes"] = [(ep.text, score) for ep, score in self.recall_episodes(query, k)]
        if mode in ("facts", "both"):
            result["facts"] = [(f"{f.key}: {f.value}", score) for f, score in self.recall_facts(query, k)]
        return result

    # --- Maintenance ---

    def decay(self, episode_half_life: float = 7.0, fact_half_life: float = 30.0) -> Dict[str, int]:
        """
        Apply importance decay to both stores. Removes memories that have
        faded below the threshold. Returns counts of pruned items.
        """
        episodes_pruned = self.episodic.prune_by_importance(
            threshold=0.1, half_life_days=episode_half_life
        )
        facts_pruned = self.semantic.prune_by_confidence(
            threshold=0.1, half_life_days=fact_half_life
        )
        return {"episodes_pruned": episodes_pruned, "facts_pruned": facts_pruned}

    def consolidate(self) -> Dict[str, int]:
        """Merge redundant facts and rebuild indexes."""
        facts_merged = self.semantic.consolidate()
        self.episodic.rebuild_index(self.embedder)
        return {"facts_merged": facts_merged}

    # --- Persistence ---

    def save(self, path: Optional[str] = None) -> None:
        """Save all memory to a JSON file."""
        save_path = path or self._persist_path
        if not save_path:
            raise ValueError("No save path provided. Pass one or set persist_path on init.")

        data = {
            "version": 1,
            "saved_at": time.time(),
            "episodic": self.episodic.to_dict(),
            "semantic": self.semantic.to_dict()
        }

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> bool:
        """Load memory from a JSON file. Returns True if successful."""
        load_path = path or self._persist_path
        if not load_path or not os.path.exists(load_path):
            return False

        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.episodic = EpisodicMemory.from_dict(data.get("episodic", {}))
        self.semantic = SemanticMemory.from_dict(data.get("semantic", {}))

        # Rebuild FAISS indexes after loading
        self.episodic.rebuild_index(self.embedder)
        self.semantic._rebuild_index()

        return True

    # --- Diagnostics ---

    def stats(self) -> Dict[str, Any]:
        """Combined stats from both stores."""
        return {
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats()
        }

    def summary(self) -> str:
        """Human-readable summary of memory state."""
        e_stats = self.episodic.stats()
        s_stats = self.semantic.stats()
        lines = [
            "Long-Term Memory Summary",
            "========================",
            f"  Episodes: {e_stats.get('total_episodes', 0)} stored, "
            f"avg importance: {e_stats.get('avg_importance', 0.0)}",
            f"  Facts:    {s_stats.get('total_facts', 0)} stored, "
            f"avg confidence: {s_stats.get('avg_confidence', 0.0)}",
        ]
        return "\n".join(lines)


# ==============================================================================
# SECTION 5 -- Demo / Self-Test
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Long-Term Memory Demo (Notebook 11)")
    print("=" * 60)

    # Create memory manager
    memory = LongTermMemory(persist_path="build-my-agent/memory_store.json")

    # --- Store some episodes ---
    print("\n--- Storing Episodes ---")
    episodes = [
        ("User asked me to calculate 987*654. I used the python_code tool. Result: 645,498.", 0.9),
        ("User asked about the capital of France. I searched the KB. Found: Paris.", 0.7),
        ("User requested a report on climate change. I searched web and found 12 articles.", 0.8),
        ("User asked me to write a Python function to sort a list. I provided bubble sort.", 0.5),
        ("The speed of light is 299,792,458 m/s. User asked about physics constants.", 0.9),
    ]
    for text, importance in episodes:
        memory.store_episode(text, importance)
        print(f"  Stored episode (importance={importance})")

    # --- Store some facts ---
    print("\n--- Storing Facts ---")
    facts = [
        ("speed_of_light", "299,792,458 meters per second in vacuum", "science"),
        ("capital_of_france", "Paris is the capital and most populous city of France", "geography"),
        ("python_sort", "Python lists have a built-in .sort() method (Timsort algorithm)", "programming"),
        ("water_boiling", "Water boils at 100 degrees Celsius (212 Fahrenheit) at sea level", "science"),
    ]
    for key, value, source in facts:
        memory.store_fact(key, value, source)
        print(f"  Stored fact: {key}")

    # --- Recall by semantic similarity ---
    print("\n--- Recall: 'What is the capital of France?' ---")
    results = memory.recall("What is the capital of France?", mode="both", k=2)
    print(f"  Episodes: {results.get('episodes', [])}")
    print(f"  Facts:    {results.get('facts', [])}")

    print("\n--- Recall: 'How fast does light travel?' ---")
    results = memory.recall("How fast does light travel?", mode="both", k=2)
    print(f"  Episodes: {results.get('episodes', [])}")
    print(f"  Facts:    {results.get('facts', [])}")

    # --- Persistence ---
    print("\n--- Saving to disk ---")
    memory.save()
    print(f"  Saved to {memory._persist_path}")

    # --- Create new instance and load ---
    print("\n--- Loading from disk ---")
    memory2 = LongTermMemory(persist_path="build-my-agent/memory_store.json")
    loaded = memory2.load()
    print(f"  Load successful: {loaded}")
    print(f"  Episodes after load: {memory2.episodic.stats()['total_episodes']}")
    print(f"  Facts after load: {memory2.semantic.stats()['total_facts']}")

    # --- Recall again after reload ---
    print("\n--- Recall after reload: 'sorting a list' ---")
    results = memory2.recall("How do I sort a list in Python?", mode="both", k=2)
    print(f"  Episodes: {results.get('episodes', [])}")
    print(f"  Facts:    {results.get('facts', [])}")

    # --- Full summary ---
    print("\n--- Memory Summary ---")
    print(memory2.summary())

    # --- Importance decay demo ---
    print("\n--- Decay Demo ---")
    # Manually set an old timestamp to simulate aging
    if memory2.episodic.episodes:
        old_episode = memory2.episodic.episodes[3]  # The sorting episode (importance=0.5)
        old_episode.timestamp = time.time() - (14 * 86400)  # 14 days old
        print(f"  Episode before decay: '{old_episode.text[:60]}...' importance={old_episode.importance}")
        print(f"  After 14 days (half_life=7): decayed={old_episode.decayed_importance(7.0):.3f}")

    pruned = memory2.decay(episode_half_life=7.0, fact_half_life=30.0)
    print(f"  Pruned: {pruned}")
    print(f"  Remaining episodes: {memory2.episodic.stats()['total_episodes']}")

    # --- Consolidation demo ---
    print("\n--- Consolidation Demo ---")
    # Add a duplicate-ish fact
    memory2.store_fact("python_sorting", "Python's list.sort() uses Timsort", "programming")
    merged = memory2.consolidate()
    print(f"  Facts merged: {merged}")
    print(f"  Total facts after consolidation: {memory2.semantic.stats()['total_facts']}")

    print("\n--- Final Summary ---")
    print(memory2.summary())

    # Clean up
    if os.path.exists("build-my-agent/memory_store.json"):
        os.remove("build-my-agent/memory_store.json")

    print("\n[OK] Long-Term Memory demo complete!")
