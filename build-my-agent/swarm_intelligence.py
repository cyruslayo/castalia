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
from agent_loop import call_llm
from parser import parse_response

# --- Utility Functions ---

def extract_ideas(text: str) -> List[str]:
    """Extract distinct ideas from text using the LLM."""
    # If the text is already short and looks like a single idea, return it
    if len(text.strip()) < 100 and "\n" not in text:
        return [text.strip()]

    messages = [
        {"role": "system", "content": "Extract distinct ideas from the text. Return each idea on a new line prefixed with '- '. Maximum 5 ideas. Be concise. Do not include preamble or chatty text."},
        {"role": "user", "content": f"Text to analyze:\n{text}"}
    ]
    result = call_llm(messages)
    
    # Extract lines starting with '-'
    ideas = [line.strip().lstrip("- ").strip() for line in result.split("\n") if line.strip().startswith("-")]
    
    # Fallback: if no bullet points, try to find lines that look like ideas
    if not ideas:
        lines = [l.strip() for l in result.split("\n") if l.strip() and not l.lower().startswith(("here", "sure", "ok", "extract"))]
        ideas = lines[:5]
            
    # Final fallback: return the original text truncated
    if not ideas:
        ideas = [text.strip()[:200]]
            
    return [i for i in ideas if i]

def rate_idea(idea: str, topic: str) -> float:
    """Rate an idea's relevance and quality (0-1) using the LLM."""
    messages = [
        {"role": "system", "content": "Rate this idea on a scale of 1-10 for relevance and originality. Reply with ONLY a number."},
        {"role": "user", "content": f"Topic: {topic}\nIdea: {idea}"}
    ]
    result = call_llm(messages)
    try:
        # Find the first number in the response
        match = re.search(r'(\d+\.?\d*)', result)
        if match:
            score = float(match.group(1))
            return min(score / 10.0, 1.0)
        return 0.5
    except:
        return 0.5

# --- Core Classes ---

class SimpleSwarmAgent:
    """
    A simple swarm agent that explores, shares, and builds on ideas.
    
    Each agent has a persona and maintains its own history of ideas 
    and discoveries received from neighbors.
    """

    def __init__(self, agent_id: int, persona: str = ""):
        self.agent_id = agent_id
        self.persona = persona or f"creative thinker #{agent_id}"
        self.ideas: List[Dict[str, Any]] = []
        self.received: List[Dict[str, Any]] = []
        self.best_idea: Optional[Dict[str, Any]] = None
        self.generation_count = 0

    def explore(self, topic: str, angle: str = "") -> List[str]:
        """Generate new ideas about a topic, optionally building on received context."""
        self.generation_count += 1
        
        prompt = f"Brainstorm 2-3 creative and specific ideas about: {topic}"
        if angle:
            prompt += f"\nFocus on this specific angle: {angle}"
            
        # If we've received ideas from neighbors, use them as context
        if self.received:
            # Take the 2 most recent discoveries
            recent = self.received[-2:]
            context = "\n".join([f"- {r['idea']}" for r in recent])
            prompt += f"\nBuild on (but don't repeat) these related ideas from your neighbors:\n{context}"

        messages = [
            {"role": "system", "content": f"You are {self.persona}. Generate novel, specific ideas. Each idea should be 1-2 sentences. Be creative and practical."},
            {"role": "user", "content": prompt}
        ]
        
        response = call_llm(messages)
        new_idea_texts = extract_ideas(response)

        new_entries = []
        for idea_text in new_idea_texts:
            entry = {
                "idea": idea_text, 
                "source": f"agent_{self.agent_id}", 
                "gen": self.generation_count,
                "persona": self.persona
            }
            self.ideas.append(entry)
            new_entries.append(idea_text)
            
            # Simple heuristic for "best": longest idea (often most detailed)
            if self.best_idea is None or len(idea_text) > len(self.best_idea.get("idea", "")):
                self.best_idea = entry

        return new_entries

    def share_discovery(self) -> Optional[Dict[str, Any]]:
        """Share the best idea discovered so far."""
        return self.best_idea

    def receive_discovery(self, discovery: Dict[str, Any]):
        """Receive and store a discovery from a neighbor."""
        # Don't receive our own discoveries
        if discovery and discovery["source"] != f"agent_{self.agent_id}":
            # Avoid duplicates
            if not any(d["idea"] == discovery["idea"] for d in self.received):
                self.received.append(discovery)

    def build_on(self, idea_text: str, topic: str) -> str:
        """Explicitly extend or improve an existing idea."""
        self.generation_count += 1
        messages = [
            {"role": "system", "content": f"You are {self.persona}. Take this idea and improve it or extend it in a novel direction. Be specific and concise (2-3 sentences)."},
            {"role": "user", "content": f"Topic: {topic}\nOriginal idea: {idea_text}\nYour improved/extended version:"}
        ]
        result = call_llm(messages)
        improved_text = result.strip()
        
        improved_entry = {
            "idea": improved_text, 
            "source": f"agent_{self.agent_id}", 
            "gen": self.generation_count, 
            "built_on": idea_text[:50],
            "persona": self.persona
        }
        self.ideas.append(improved_entry)
        
        if self.best_idea is None or len(improved_text) > len(self.best_idea.get("idea", "")):
            self.best_idea = improved_entry
            
        return improved_text

    def __repr__(self):
        return f"SimpleSwarmAgent(id={self.agent_id}, ideas={len(self.ideas)}, received={len(self.received)})"


class SwarmCoordinator:
    """
    Manages a swarm of agents with a local communication topology.
    
    Default topology: Ring (each agent connects to k neighbors).
    """

    def __init__(self, n_agents: int, k_neighbors: int = 2):
        self.agents = [SimpleSwarmAgent(i, f"specialist #{i}") for i in range(n_agents)]
        self.k = min(k_neighbors, n_agents - 1)
        self.round_log: List[Dict[str, Any]] = []
        self.all_ideas: List[Dict[str, Any]] = []

    def _get_neighbors(self, agent_idx: int) -> List[int]:
        """Ring topology: each agent connects to k nearest neighbors."""
        n = len(self.agents)
        neighbors = []
        # Connect to neighbors on both sides
        for offset in range(1, (self.k // 2) + 1):
            neighbors.append((agent_idx + offset) % n)
            neighbors.append((agent_idx - offset) % n)
        
        # If k is odd, add one more
        if self.k % 2 != 0:
            neighbors.append((agent_idx + (self.k // 2) + 1) % n)
            
        return list(set(neighbors))[:self.k]

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
            print(f"  Agent {i}: explored -> {len(ideas)} ideas")

        # Phase 2: Share & Receive (local topology)
        for i, agent in enumerate(self.agents):
            discovery = agent.share_discovery()
            if discovery:
                neighbors = self._get_neighbors(i)
                for neighbor_idx in neighbors:
                    self.agents[neighbor_idx].receive_discovery(discovery)
        
        # Record stats
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
