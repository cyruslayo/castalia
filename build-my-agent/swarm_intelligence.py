"""
Swarm Intelligence - Emergent behavior from simple agents.

This module implements a swarm of simple agents that interact locally to produce 
emergent global behavior, inspired by ant foraging and bee waggle dances.

Architecture:
1. SimpleSwarmAgent: A basic agent that explores, shares, and builds on ideas.
2. SwarmCoordinator: Manages the topology (ring) and coordinates rounds of interaction.
"""

import time
import re
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

# Import core infrastructure
from config import get_model, get_client

# --- Utility Functions ---

def _llm_enabled() -> bool:
    """Return True when swarm demos should call the external LLM service."""
    # Unit tests and examples should be deterministic/offline by default. Set this
    # for interactive demos that intentionally exercise the configured endpoint.
    import os
    return os.environ.get("SWARM_USE_LLM", "").lower() in {"1", "true", "yes"}


def _call_swarm_llm(messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Call the configured LLM with swarm-sized token limits."""
    response = get_client().chat.completions.create(
        model=get_model(),
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=60,
    )
    message = response.choices[0].message
    return message.content or getattr(message, "reasoning", "") or ""


def extract_ideas(text: str) -> List[str]:
    """Extract distinct ideas from text deterministically.

    The swarm tests are about coordination mechanics, not endpoint availability, so
    extraction must not block on the remote LLM. This parser handles bullets,
    numbered lists, and prose such as "first idea ... second idea ...".
    """
    if not text:
        return []

    cleaned = text.strip()
    if len(cleaned) < 100 and "\n" not in cleaned:
        return [cleaned]

    ideas: List[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        match = re.match(r"^(?:[-*•]|\d+[.)])\s*(.+)$", line)
        if match:
            ideas.append(match.group(1).strip())

    if not ideas:
        normalized = re.sub(r"\s+", " ", cleaned)
        parts = re.split(
            r"(?i)(?:\bthe\s+)?(?:first|second|third|fourth|fifth|another|next)\s+idea\s+(?:is\s+to|is|involves|would|could)?\s*",
            normalized,
        )
        ideas = [part.strip(" .;:") for part in parts if part.strip(" .;:")]

    if not ideas:
        ideas = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]

    # De-duplicate while preserving order.
    distinct: List[str] = []
    seen = set()
    for idea in ideas:
        key = idea.lower()
        if key not in seen:
            seen.add(key)
            distinct.append(idea)
    return distinct[:5]


def rate_idea(idea: str, topic: str) -> float:
    """Rate an idea's relevance and quality (0-1) without requiring the LLM."""
    idea_l = idea.lower()
    topic_terms = {t for t in re.findall(r"[a-z]{4,}", topic.lower())}
    idea_terms = {t for t in re.findall(r"[a-z]{4,}", idea_l)}

    score = 0.45
    if topic_terms:
        score += min(len(topic_terms & idea_terms) / len(topic_terms), 1.0) * 0.35
    if any(word in idea_l for word in ("ai", "data", "sensor", "satellite", "optimize", "renewable", "energy", "detect")):
        score += 0.15
    if any(word in idea_l for word in ("magic", "instant", "impossible")):
        score -= 0.3
    return max(0.0, min(score, 1.0))

# --- Core Classes ---

class SimpleSwarmAgent:
    """
    A simple swarm agent that explores, shares, and builds on ideas.
    
    2026 SOTA Upgrade: Supports heterogeneous roles (explorer, skeptic, verifier)
    and implements a verification gate to prevent hallucination cascades.
    """

    def __init__(self, agent_id: int, role: str = "explorer", persona: str = ""):
        self.agent_id = agent_id
        self.role = role.lower()
        default_persona = f"creative thinker #{agent_id}" if role == "explorer" else f"critical analyst #{agent_id}"
        self.persona = persona or default_persona
        self.ideas: List[Dict[str, Any]] = []
        self.received: List[Dict[str, Any]] = []
        self.best_idea: Optional[Dict[str, Any]] = None
        self.generation_count = 0

    def explore(self, topic: str, angle: str = "") -> List[str]:
        """Generate new ideas about a topic, optionally building on received context."""
        self.generation_count += 1
        
        if self.role == "skeptic":
            prompt = f"Critically analyze the topic: {topic}. Find 2-3 hidden risks, edge cases, or common misconceptions."
        else:
            prompt = f"Brainstorm 2-3 creative and specific ideas about: {topic}"
            
        if angle:
            prompt += f"\nFocus on this specific angle: {angle}"
            
        if self.received:
            # Take the 2 most recent discoveries
            recent = self.received[-2:]
            context = "\n".join([f"- {r['idea']} (Confidence: {r.get('score', 'N/A')})" for r in recent])
            
            if self.role == "skeptic":
                prompt += f"\nIdentify flaws or logic gaps in these ideas from your neighbors:\n{context}"
            else:
                prompt += f"\nBuild on (but don't repeat) these related ideas from your neighbors:\n{context}"

        if _llm_enabled():
            messages = [
                {"role": "system", "content": f"You are {self.persona} (Role: {self.role}). Generate novel, specific points. Each point should be 1-2 sentences. Be {'critical and precise' if self.role == 'skeptic' else 'creative and practical'}."},
                {"role": "user", "content": prompt}
            ]
            response = _call_swarm_llm(messages, max_tokens=220, temperature=0.7)
            new_idea_texts = extract_ideas(response)
        else:
            new_idea_texts = self._local_ideas(topic, angle)

        new_entries = []
        for idea_text in new_idea_texts:
            score = rate_idea(idea_text, topic)
            entry = {
                "idea": idea_text, 
                "source": f"agent_{self.agent_id}", 
                "gen": self.generation_count,
                "persona": self.persona,
                "role": self.role,
                "score": score
            }
            self.ideas.append(entry)
            new_entries.append(idea_text)
            
            # Update best idea based on LLM score
            if self.best_idea is None or score > self.best_idea.get("score", 0):
                self.best_idea = entry

        return new_entries

    def verify_discovery(self, discovery: Dict[str, Any]) -> float:
        """
        2026 SOTA: RLVR-Lite verification gate.
        Checks if a neighbor's discovery is logically sound.
        """
        idea = discovery.get("idea", "")
        if not _llm_enabled():
            return rate_idea(idea, "")

        messages = [
            {"role": "system", "content": "You are a verification gate. Rate the logical consistency and factual plausibility of this idea on a scale of 0.0 to 1.0. Reply ONLY with the number."},
            {"role": "user", "content": f"Idea: {idea}"}
        ]
        result = _call_swarm_llm(messages, max_tokens=20, temperature=0.1)
        try:
            match = re.search(r'(\d+\.?\d*)', result)
            return float(match.group(1)) if match else 0.5
        except:
            return 0.5

    def _local_ideas(self, topic: str, angle: str = "") -> List[str]:
        """Generate deterministic placeholder ideas for offline tests/demos."""
        focus = angle or ("risks" if self.role == "skeptic" else "implementation")
        if self.role == "skeptic":
            return [
                f"Stress-test {topic} for grid reliability, cost overruns, and maintenance bottlenecks around {focus}.",
                f"Map adoption risks for {topic}, especially incentives, regulation, and edge-case failures.",
            ]
        return [
            f"Use data-driven pilots to improve {topic} with measurable milestones around {focus}.",
            f"Create community-scale prototypes for {topic} that combine sensors, forecasting, and transparent feedback loops.",
        ]

    def share_discovery(self) -> Optional[Dict[str, Any]]:
        """Share the best idea discovered so far."""
        return self.best_idea

    def receive_discovery(self, discovery: Dict[str, Any], threshold: float = 0.4):
        """
        Receive and store a discovery from a neighbor if it passes the verification gate.
        """
        if not discovery or discovery["source"] == f"agent_{self.agent_id}":
            return

        # Verification Gate
        v_score = self.verify_discovery(discovery)
        if v_score >= threshold:
            # Avoid duplicates
            if not any(d["idea"] == discovery["idea"] for d in self.received):
                discovery_copy = discovery.copy()
                discovery_copy["v_score"] = v_score
                self.received.append(discovery_copy)

    def build_on(self, idea_text: str, topic: str) -> str:
        """Explicitly extend or improve an existing idea."""
        self.generation_count += 1
        if _llm_enabled():
            messages = [
                {"role": "system", "content": f"You are {self.persona}. Take this idea and improve it or extend it in a novel direction. Be specific and concise (2-3 sentences)."},
                {"role": "user", "content": f"Topic: {topic}\nOriginal idea: {idea_text}\nYour improved/extended version:"}
            ]
            result = _call_swarm_llm(messages, max_tokens=180, temperature=0.7)
            improved_text = result.strip()
        else:
            improved_text = f"Extend '{idea_text}' for {topic} by adding a small pilot, measurable success criteria, and a feedback loop."
        score = rate_idea(improved_text, topic)
        
        improved_entry = {
            "idea": improved_text, 
            "source": f"agent_{self.agent_id}", 
            "gen": self.generation_count, 
            "built_on": idea_text[:50],
            "persona": self.persona,
            "role": self.role,
            "score": score
        }
        self.ideas.append(improved_entry)
        
        if self.best_idea is None or score > self.best_idea.get("score", 0):
            self.best_idea = improved_entry
            
        return improved_text

    def __repr__(self):
        return f"SimpleSwarmAgent(id={self.agent_id}, role={self.role}, ideas={len(self.ideas)}, received={len(self.received)})"


class SwarmCoordinator:
    """
    Manages a swarm of agents with a local communication topology.
    
    2026 SOTA Upgrade: Supports Heterogeneous roles and Dynamic Topology (AOV-Lite).
    """

    def __init__(self, n_agents: int, k_neighbors: int = 2, skeptic_ratio: float = 0.4):
        self.agents = []
        for i in range(n_agents):
            role = "skeptic" if (i / n_agents) < skeptic_ratio else "explorer"
            self.agents.append(SimpleSwarmAgent(i, role=role))
            
        self.k = min(k_neighbors, n_agents - 1)
        self.round_log: List[Dict[str, Any]] = []
        self.all_ideas: List[Dict[str, Any]] = []

    def _get_neighbors(self, agent_idx: int, k_override: int = None) -> List[int]:
        """Ring topology: each agent connects to k nearest neighbors."""
        n = len(self.agents)
        k = k_override if k_override is not None else self.k
        k = min(k, n - 1)
        
        neighbors = []
        for offset in range(1, (k // 2) + 1):
            neighbors.append((agent_idx + offset) % n)
            neighbors.append((agent_idx - offset) % n)
        
        if k % 2 != 0:
            neighbors.append((agent_idx + (k // 2) + 1) % n)
            
        return list(set(neighbors))[:k]

    def run_round(self, topic: str, round_num: int, angles: List[str] = None) -> Dict[str, Any]:
        """
        Execute one round of the swarm interaction:
        1. Explore: Agents generate ideas independently (Phase 1)
        2. Share & Receive: Agents broadcast best ideas to neighbors (Phase 2)
        """
        round_idea_texts = []
        print(f"\n--- Swarm Round {round_num} ---")

        # Phase 1: Explore
        for i, agent in enumerate(self.agents):
            angle = angles[i] if angles and i < len(angles) else ""
            ideas = agent.explore(topic, angle)
            round_idea_texts.extend(ideas)
            print(f"  Agent {i} ({agent.role}): explored -> {len(ideas)} points")

        # Phase 2: Share & Receive (Dynamic Topology)
        for i, agent in enumerate(self.agents):
            discovery = agent.share_discovery()
            if discovery:
                score = discovery.get("score", 0.5)
                # Dynamic Sharing: High-scoring ideas spread further
                k_effective = self.k + 2 if score > 0.8 else self.k
                neighbors = self._get_neighbors(i, k_override=k_effective)
                
                for neighbor_idx in neighbors:
                    self.agents[neighbor_idx].receive_discovery(discovery)
        
        round_info = {
            "round": round_num,
            "total_ideas": len(round_idea_texts),
            "unique_ideas": len(set(round_idea_texts)),
        }
        self.round_log.append(round_info)
        return round_info

    def run_swarm(self, topic: str, n_rounds: int, angles: List[str] = None) -> Dict[str, Any]:
        """Run the full swarm process for multiple rounds."""
        start_time = time.time()
        print(f"Starting Swarm: {len(self.agents)} agents, {n_rounds} rounds, k={self.k} neighbors")
        print(f"Topic: {topic}")

        for r in range(n_rounds):
            self.run_round(topic, r + 1, angles)

        # Collect all unique ideas
        all_idea_texts = []
        for agent in self.agents:
            for entry in agent.ideas:
                all_idea_texts.append(entry["idea"])

        unique_ideas = list(set(all_idea_texts))
        elapsed = time.time() - start_time

        result = {
            "topic": topic,
            "total_ideas": len(all_idea_texts),
            "unique_ideas": len(unique_ideas),
            "ideas_per_agent": {f"agent_{a.agent_id}": len(a.ideas) for a in self.agents},
            "time_seconds": round(elapsed, 2),
            "rounds": self.round_log,
            "all_unique": unique_ideas,
            "agents": [repr(a) for a in self.agents]
        }
        return result

# --- Main (Self-Test) ---

if __name__ == "__main__":
    topic = "Innovative ways to reduce food waste using technology"
    angles = [
        "consumer/household level solutions",
        "supply chain and logistics",
        "AI and data analytics approaches",
        "community and social platforms",
        "hardware and IoT sensors"
    ]
    
    coordinator = SwarmCoordinator(n_agents=5, k_neighbors=2)
    results = coordinator.run_swarm(topic, n_rounds=2, angles=angles)
    
    print("\n" + "=" * 60)
    print("SWARM EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Total ideas generated: {results['total_ideas']}")
    print(f"Unique ideas: {results['unique_ideas']}")
    print(f"Time: {results['time_seconds']}s")
    print("\nSample Ideas:")
    for i, idea in enumerate(results['all_unique'][:10], 1):
        print(f"{i}. {idea}")
