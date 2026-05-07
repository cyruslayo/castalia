"""
Knowledge Graph Memory for Agents (Notebook 12)

Structured memory that stores facts as (subject, relation, object) triples
using networkx. Complements vector-based long-term memory by enabling
multi-hop reasoning and relationship traversal.

Key components:
  - extract_triples()     : LLM-based entity-relation extraction
  - GraphMemory           : networkx DiGraph with canonical entities
  - LLMQueryClassifier    : Option B -- LLM-based intent classification
  - HybridMemory          : vector + graph with LLM-routed queries
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
import faiss
import networkx as nx

from config import get_client, get_model

# ==============================================================================
# SECTION 0 -- LLM Helper (matches pattern in agent_loop.py)
# ==============================================================================


def _llm_generate(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Send messages to the LLM and return the response text."""
    client = get_client()
    model = get_model()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    # Hermes-model quirk: content may be None, reasoning field may hold text
    if content is None or content.strip() == "":
        reasoning = getattr(response.choices[0].message, "reasoning", None)
        if reasoning:
            content = reasoning
        else:
            content = ""
    return content.strip()


# ==============================================================================
# SECTION 1 -- Triple Extraction
# ==============================================================================


def extract_triples(text: str, max_triples: int = 10) -> List[Tuple[str, str, str]]:
    """
    Extract (subject, relation, object) triples from text using the LLM.

    Returns a list of (subject, relation, object) tuples.
    Relations are normalized to lowercase_snake_case.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at extracting structured knowledge from text.\n"
                "Extract (subject, relation, object) triples from the given text.\n"
                "\n"
                "Rules:\n"
                "- Normalize entity names (e.g., \"Dr. Smith\" -> \"Dr Smith\")\n"
                "- Use lowercase_snake_case for relations (e.g., works_at, is_a, manages)\n"
                "- Extract concrete facts, not opinions or speculation\n"
                "- One triple per line in format: subject | relation | object\n"
                "- Output ONLY triples, no commentary"
            ),
        },
        {
            "role": "user",
            "content": f"Extract triples from:\n{text}",
        },
    ]

    response = _llm_generate(messages, max_tokens=300, temperature=0.2)

    triples = []
    for line in response.strip().split("\n"):
        line = line.strip().lstrip("- 0123456789.")
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(p for p in parts):
            subject, relation, obj = parts
            triples.append((subject, relation.lower().strip().replace(" ", "_"), obj))

    return triples[:max_triples]


# ==============================================================================
# SECTION 2 -- GraphMemory
# ==============================================================================


class GraphMemory:
    """
    Knowledge graph memory using networkx for structured fact storage.

    Stores facts as directed edges: subject --[relation]--> object.
    Entities are normalized to a canonical form to prevent duplicates
    like "Alice" and "alice" from becoming separate nodes.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_index: Dict[str, str] = {}  # normalized -> canonical
        self.triple_count = 0

    # --- entity normalization ---

    def _normalize(self, entity: str) -> str:
        """Normalize entity name for matching."""
        return entity.lower().strip()

    def _canonical(self, entity: str) -> str:
        """Get or create canonical name for an entity."""
        normalized = self._normalize(entity)
        if normalized not in self.entity_index:
            self.entity_index[normalized] = entity.strip()
        return self.entity_index[normalized]

    # --- core graph operations ---

    def add_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a single (subject, relation, object) triple to the graph.

        Returns True if a new edge was added, False if duplicate.
        """
        s = self._canonical(subject)
        o = self._canonical(obj)
        r = relation.lower().strip()

        self.graph.add_node(s, type="entity")
        self.graph.add_node(o, type="entity")

        edge_data = {"relation": r, "added_at": time.time()}
        if metadata:
            edge_data.update(metadata)

        # Duplicate check: same subject + relation + object
        if self.graph.has_edge(s, o):
            existing = self.graph[s][o]
            if existing.get("relation") == r:
                return False

        self.graph.add_edge(s, o, **edge_data)
        self.triple_count += 1
        return True

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[int, List[Tuple[str, str, str]]]:
        """Extract triples from text and add them to the graph."""
        triples = extract_triples(text)
        added = 0
        for s, r, o in triples:
            if self.add_triple(s, r, o, metadata):
                added += 1
        return added, triples

    # --- querying ---

    def get_entity_context(self, entity: str) -> List[Dict[str, Any]]:
        """Get all facts about an entity (incoming and outgoing edges)."""
        canonical = self._canonical(entity)
        if canonical not in self.graph:
            # Try fuzzy match: normalized query appears inside any node name
            normalized = self._normalize(entity)
            for node in self.graph.nodes():
                if normalized in self._normalize(node):
                    canonical = node
                    break
            else:
                return []

        facts = []
        # Outgoing: entity -> relation -> other
        for _, target, data in self.graph.out_edges(canonical, data=True):
            facts.append(
                {
                    "subject": canonical,
                    "relation": data.get("relation", "related_to"),
                    "object": target,
                    "direction": "outgoing",
                }
            )
        # Incoming: other -> relation -> entity
        for source, _, data in self.graph.in_edges(canonical, data=True):
            facts.append(
                {
                    "subject": source,
                    "relation": data.get("relation", "related_to"),
                    "object": canonical,
                    "direction": "incoming",
                }
            )
        return facts

    def multi_hop_query(self, entity: str, hops: int = 2) -> Dict[str, Any]:
        """
        Traverse N hops from an entity, collecting all connected facts.

        Uses BFS (Breadth-First Search) to find shortest paths first.
        Returns dict with entity, facts (each tagged with hop count),
        and the set of entities reached.
        """
        canonical = self._canonical(entity)
        if canonical not in self.graph:
            return {"entity": entity, "facts": [], "entities_reached": set(), "hops": hops}

        visited: Set[str] = set()
        facts: List[Dict[str, Any]] = []
        frontier: Set[str] = {canonical}

        for hop in range(hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)

                # Outgoing edges
                for _, target, data in self.graph.out_edges(node, data=True):
                    facts.append(
                        {
                            "subject": node,
                            "relation": data.get("relation", "related_to"),
                            "object": target,
                            "hop": hop + 1,
                        }
                    )
                    next_frontier.add(target)

                # Incoming edges
                for source, _, data in self.graph.in_edges(node, data=True):
                    facts.append(
                        {
                            "subject": source,
                            "relation": data.get("relation", "related_to"),
                            "object": node,
                            "hop": hop + 1,
                        }
                    )
                    next_frontier.add(source)

            frontier = next_frontier - visited

        return {
            "entity": entity,
            "facts": facts,
            "entities_reached": visited,
            "hops": hops,
        }

    def query(self, question: str) -> List[Dict[str, Any]]:
        """
        Find relevant graph context for a natural language question.

        1. LLM extracts entity names from the question
        2. Looks up each entity's context
        3. Falls back to scanning all nodes if no entities found
        4. Deduplicates results
        """
        # Step 1: LLM entity extraction
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract entity names from the user's question.\n"
                    "Output one entity per line, nothing else.\n"
                    "If no specific entities are mentioned, output 'NONE'."
                ),
            },
            {"role": "user", "content": question},
        ]
        response = _llm_generate(messages, max_tokens=100, temperature=0.2)
        entities = [
            e.strip().lstrip("- 0123456789.")
            for e in response.strip().split("\n")
            if e.strip() and e.strip().upper() != "NONE"
        ]

        # Step 2: Look up each entity
        all_facts: List[Dict[str, Any]] = []
        for entity in entities:
            context = self.get_entity_context(entity)
            all_facts.extend(context)

        # Step 3: Fallback -- scan all nodes for partial string match
        if not all_facts:
            q_lower = question.lower()
            for node in self.graph.nodes():
                if self._normalize(node) in q_lower or q_lower in self._normalize(node):
                    context = self.get_entity_context(node)
                    all_facts.extend(context)

        # Step 4: Deduplicate by (subject, relation, object)
        seen: Set[Tuple[str, str, str]] = set()
        unique_facts: List[Dict[str, Any]] = []
        for f in all_facts:
            key = (f["subject"], f["relation"], f["object"])
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)

        return unique_facts

    # --- diagnostics ---

    def stats(self) -> Dict[str, Any]:
        return {
            "entities": self.graph.number_of_nodes(),
            "relations": self.graph.number_of_edges(),
            "triples_added": self.triple_count,
        }

    def visualize(self, max_nodes: int = 20) -> str:
        """ASCII visualization of the knowledge graph."""
        lines = [
            f"Knowledge Graph: {self.graph.number_of_nodes()} entities, {self.graph.number_of_edges()} relations",
            "=" * 60,
        ]
        shown = 0
        for node in sorted(self.graph.nodes()):
            out_edges = list(self.graph.out_edges(node, data=True))
            if out_edges:
                lines.append(f"\n  * {node}")
                for _, target, data in out_edges:
                    rel = data.get("relation", "?")
                    lines.append(f"     --[{rel}]--> {target}")
                shown += 1
                if shown >= max_nodes:
                    remaining = self.graph.number_of_nodes() - shown
                    lines.append(f"\n  ... ({remaining} more entities)")
                    break
        return "\n".join(lines)

    def __repr__(self) -> str:
        s = self.stats()
        return f"GraphMemory({s['entities']} entities, {s['relations']} relations)"


# ==============================================================================
# SECTION 3 -- LLM Query Classifier (Option B)
# ==============================================================================


class LLMQueryClassifier:
    """
    LLM-based query intent classifier for hybrid memory routing.

    Replaces hardcoded keyword matching with a lightweight LLM call that
    classifies each query as SEMANTIC, RELATIONAL, or BOTH.

    Design choices:
      - max_tokens=5: classification is a single word, minimal cost
      - temperature=0.0: deterministic, no creativity needed
      - fallback to BOTH: if LLM returns garbage, safe default
    """

    SYSTEM_PROMPT = (
        "You are a query routing classifier.\n"
        "Classify the user's question into EXACTLY ONE category:\n"
        "\n"
        "  SEMANTIC   -- asks for information ABOUT a topic\n"
        "                (e.g., 'Tell me about databases', 'Describe cloud infra')\n"
        "                Includes: 'what is/are' questions seeking descriptions\n"
        "\n"
        "  RELATIONAL -- asks about CONNECTIONS between specific entities\n"
        "                (e.g., 'Who manages the backend team?', 'What connects Alice to Kubernetes?')\n"
        "                Includes: questions with explicit entity names and relationship verbs\n"
        "\n"
        "Respond with EXACTLY one word -- SEMANTIC or RELATIONAL.\n"
        "NEVER say BOTH. NEVER explain. Output ONE WORD ONLY."
    )

    def __init__(self, max_tokens: int = 5, temperature: float = 0.0):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._cache: Dict[str, str] = {}  # question -> classification cache

    def classify(self, question: str) -> str:
        """
        Classify a query. Returns 'semantic', 'relational', or 'both'.
        """
        # Check cache first
        cache_key = question.strip().lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        response = _llm_generate(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # Step 1: LLM classification
        llm_result = self._classify_with_llm(response)

        # Step 2: Keyword heuristic (parallel signal)
        keyword_result = self._keyword_classify(question)

        # Step 3: Confidence-weighted decision
        if llm_result == keyword_result:
            # Both agree — high confidence
            result = llm_result
        elif keyword_result == "semantic" and llm_result == "relational":
            # LLM says relational, keywords say semantic
            # This catches "What are X?" questions that the LLM misreads as entity-seeking
            # Keywords are better here because "what are/what is" is clearly informational
            result = "semantic"
        elif keyword_result == "relational" and llm_result == "semantic":
            # LLM says semantic, keywords say relational
            # LLM is usually right here — trust it (keywords may false-positive on common words)
            result = "semantic"
        else:
            result = llm_result  # default to LLM

        self._cache[cache_key] = result
        return result

    def _classify_with_llm(self, response: str) -> str:
        """Extract classification from LLM response."""
        words = response.strip().upper().split()
        first_word = words[0] if words else ""

        if first_word == "SEMANTIC":
            return "semantic"
        elif first_word == "RELATIONAL":
            return "relational"
        else:
            # Fallback: scan for the word anywhere in response
            if "SEMANTIC" in response.upper():
                return "semantic"
            elif "RELATION" in response.upper():
                return "relational"
            return "relational"  # default

    def _keyword_classify(self, question: str) -> str:
        """Lightweight keyword heuristic as a parallel signal."""
        q_lower = question.lower()

        # Strong semantic indicators
        semantic_kw = [
            "tell me about", "describe", "explain", "what is", "what are",
            "overview", "summary", "how does", "what does"
        ]
        # Strong relational indicators (need entity + relationship verb)
        relational_kw = [
            "who manages", "who leads", "who reports", "reports to",
            "what connects", "depends on", "chain from", "path from",
            "who works on", "relationship between"
        ]

        sem_score = sum(1 for kw in semantic_kw if kw in q_lower)
        rel_score = sum(1 for kw in relational_kw if kw in q_lower)

        if rel_score > sem_score:
            return "relational"
        elif sem_score > rel_score:
            return "semantic"
        else:
            return "relational"  # default bias toward graph (more actionable)

    def batch_classify(self, questions: List[str]) -> List[str]:
        """Classify multiple queries (uses cache, but still one LLM call per unique query)."""
        return [self.classify(q) for q in questions]

    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self._cache.clear()


# ==============================================================================
# SECTION 4 -- HybridMemory (Vector + Graph with LLM Router)
# ==============================================================================


class HybridMemory:
    """
    Combined vector store and knowledge graph with LLM-based query routing.

    When a query arrives:
      1. LLMQueryClassifier decides if it's SEMANTIC, RELATIONAL, or BOTH
      2. SEMANTIC  -> query vector store only (FAISS + embeddings)
      3. RELATIONAL -> query graph only (networkx traversal)
      4. BOTH      -> query both, merge and deduplicate

    On store(text), writes to BOTH systems simultaneously. You cannot predict
    at ingestion time which query type will be needed later.
    """

    def __init__(self, embedding_dim: int = 384):
        # Lazy import to avoid heavy model loading at import time
        self._embedding_dim = embedding_dim
        self._embedder = None

        # Vector store
        self.vector_index = faiss.IndexFlatIP(embedding_dim)
        self.vector_texts: List[str] = []

        # Knowledge graph
        self.graph = GraphMemory()

        # Query router (Option B: LLM-based)
        self.classifier = LLMQueryClassifier()

    # --- embedding lazy loader ---

    def _get_embedder(self):
        """Lazy-load the BGE embedder (matches long_term_memory.py)."""
        if self._embedder is None:
            from long_term_memory import BGEEmbedder

            self._embedder = BGEEmbedder()
        return self._embedder

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using BGE. Returns (n_texts, 384) array."""
        embedder = self._get_embedder()
        return embedder.encode_batch(texts)

    # --- storage ---

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store text in BOTH vector store and knowledge graph."""
        # Vector store
        embedding = self._embed([text])[0]
        self.vector_index.add(np.array([embedding], dtype=np.float32))
        self.vector_texts.append(text)

        # Graph store
        added, triples = self.graph.add_memory(text, metadata)
        return {
            "vector_stored": True,
            "graph_triples": len(triples),
            "graph_new": added,
        }

    def store_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Store multiple texts efficiently."""
        if not texts:
            return []

        # Batch embed for vector store
        embeddings = self._embed(texts)
        self.vector_index.add(embeddings.astype(np.float32))
        self.vector_texts.extend(texts)

        # Graph store (LLM extraction is per-text, cannot batch easily)
        results = []
        for text in texts:
            added, triples = self.graph.add_memory(text)
            results.append(
                {
                    "vector_stored": True,
                    "graph_triples": len(triples),
                    "graph_new": added,
                }
            )
        return results

    # --- query routing ---

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Route query to appropriate memory store(s) based on LLM classification.
        """
        query_type = self.classifier.classify(question)
        results = {
            "query_type": query_type,
            "vector_results": [],
            "graph_results": [],
        }

        # Vector query
        if query_type in ("semantic", "both"):
            if self.vector_index.ntotal > 0:
                q_emb = self._embed([question])[0]
                n = min(k, self.vector_index.ntotal)
                scores, indices = self.vector_index.search(
                    np.array([q_emb], dtype=np.float32), n
                )
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0:
                        results["vector_results"].append(
                            {
                                "text": self.vector_texts[idx],
                                "similarity": float(score),
                                "source": "vector",
                            }
                        )

        # Graph query
        if query_type in ("relational", "both"):
            graph_facts = self.graph.query(question)
            for f in graph_facts[:k]:
                results["graph_results"].append(
                    {
                        "triple": f"{f['subject']} {f['relation']} {f['object']}",
                        "source": "graph",
                        **f,
                    }
                )

        return results

    def format_for_prompt(self, question: str, k: int = 5) -> str:
        """Format query results for inclusion in an LLM prompt."""
        results = self.query(question, k)

        lines = [f"Memory recall (query type: {results['query_type']})"]

        if results["vector_results"]:
            lines.append("  From semantic memory:")
            for r in results["vector_results"][:k]:
                lines.append(f"    [{r['similarity']:.2f}] {r['text'][:80]}")

        if results["graph_results"]:
            lines.append("  From knowledge graph:")
            for r in results["graph_results"][:k]:
                lines.append(f"    {r['triple']}")

        return "\n".join(lines)

    # --- diagnostics ---

    def stats(self) -> Dict[str, Any]:
        return {
            "vector_entries": len(self.vector_texts),
            "graph": self.graph.stats(),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"HybridMemory(vectors={s['vector_entries']}, "
            f"entities={s['graph']['entities']}, "
            f"relations={s['graph']['relations']})"
        )


# ==============================================================================
# SECTION 5 -- GraphMaintenance
# ==============================================================================


class GraphMaintenance:
    """Tools for maintaining knowledge graph quality over time."""

    # Relations that should have exactly one object per subject
    UNIQUE_RELATIONS: Set[str] = {
        "is_cto_of",
        "is_ceo_of",
        "leads",
        "is_lead_of",
        "manages",
        "headquartered_in",
        "founded_in",
        "uses_database",
        "deployed_on",
    }

    @classmethod
    def find_contradictions(cls, graph_memory: GraphMemory) -> List[Dict[str, Any]]:
        """
        Find potential contradictions: same (subject, relation) -> multiple objects
        for relations that should be unique.
        """
        edge_groups = defaultdict(list)
        for s, o, data in graph_memory.graph.edges(data=True):
            relation = data.get("relation", "unknown")
            edge_groups[(s, relation)].append(o)

        contradictions = []
        for (subject, relation), objects in edge_groups.items():
            if len(objects) > 1 and relation in cls.UNIQUE_RELATIONS:
                contradictions.append(
                    {
                        "subject": subject,
                        "relation": relation,
                        "objects": objects,
                        "issue": f"Multiple values for unique relation: {subject} {relation} -> {objects}",
                    }
                )
        return contradictions

    @classmethod
    def graph_health_report(cls, graph_memory: GraphMemory) -> Dict[str, Any]:
        """Generate a health report for the knowledge graph."""
        g = graph_memory.graph
        stats = graph_memory.stats()

        weakly_connected = (
            nx.number_weakly_connected_components(g) if g.number_of_nodes() > 0 else 0
        )
        isolated = list(nx.isolates(g))
        degrees = [d for _, d in g.degree()] if g.number_of_nodes() > 0 else []
        avg_degree = sum(degrees) / len(degrees) if degrees else 0

        return {
            "entities": stats["entities"],
            "relations": stats["relations"],
            "connected_components": weakly_connected,
            "isolated_nodes": len(isolated),
            "avg_degree": round(avg_degree, 2),
            "max_degree": max(degrees) if degrees else 0,
        }


# ==============================================================================
# SECTION 6 -- Self-Test Demo
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Graph Memory Self-Test")
    print("=" * 60)

    # --- Test 1: Triple extraction ---
    print("\n[Test 1] Triple Extraction")
    sample = (
        "Alice is the tech lead of Project Phoenix. "
        "The project uses PostgreSQL for its database and Redis for caching. "
        "Bob handles the frontend design."
    )
    triples = extract_triples(sample)
    print(f"  Input: {sample[:60]}...")
    print(f"  Extracted {len(triples)} triples:")
    for s, r, o in triples:
        print(f"    ({s}) --[{r}]--> ({o})")

    # --- Test 2: GraphMemory build & query ---
    print("\n[Test 2] GraphMemory")
    gm = GraphMemory()
    texts = [
        "Alice is the tech lead of Project Phoenix",
        "Project Phoenix uses PostgreSQL for its database",
        "Bob is the frontend developer on Project Phoenix",
        "PostgreSQL 15 supports the MERGE statement",
    ]
    for text in texts:
        added, _ = gm.add_memory(text)
        print(f"  Stored: {text[:50]}... -> {added} new triples")

    print(f"\n  {gm}")

    # Entity context
    print("\n  Entity context for 'Alice':")
    for f in gm.get_entity_context("Alice"):
        print(f"    {f['subject']} --[{f['relation']}]--> {f['object']} ({f['direction']})")

    # Multi-hop query
    print("\n  Multi-hop query (Alice, 2 hops):")
    result = gm.multi_hop_query("Alice", hops=2)
    print(f"    Entities reached: {result['entities_reached']}")
    for f in result["facts"]:
        print(f"    [hop {f['hop']}] {f['subject']} --[{f['relation']}]--> {f['object']}")

    # Natural language query
    print("\n  NL query: 'What database does Project Phoenix use?'")
    facts = gm.query("What database does Project Phoenix use?")
    for f in facts:
        print(f"    {f['subject']} --[{f['relation']}]--> {f['object']}")

    # --- Test 3: LLMQueryClassifier ---
    print("\n[Test 3] LLMQueryClassifier (Option B)")
    classifier = LLMQueryClassifier()
    test_queries = [
        "Tell me about databases and storage",
        "Who reports to Alice?",
        "What technology does the Atlas project use?",
    ]
    for q in test_queries:
        label = classifier.classify(q)
        print(f"  '{q[:45]}...' -> {label.upper()}")

    # --- Test 4: HybridMemory (requires BGE embedder) ---
    print("\n[Test 4] HybridMemory")
    print("  (Loading BGE embedder -- this may take ~30s on first run)")
    try:
        hybrid = HybridMemory()
        for text in texts:
            result = hybrid.store(text)
            print(f"  Stored: {text[:45]}... -> {result}")

        print(f"\n  {hybrid}")

        for q in test_queries:
            results = hybrid.query(q, k=2)
            print(f"\n  Query: '{q[:45]}...'")
            print(f"    Routed to: {results['query_type']}")
            print(f"    Vector: {len(results['vector_results'])}, Graph: {len(results['graph_results'])}")
    except Exception as e:
        print(f"  Skipped (embedder not available or LLM error): {e}")

    print("\n" + "=" * 60)
    print("Self-test complete.")
    print("=" * 60)
