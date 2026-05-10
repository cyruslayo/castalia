"""
Agent Orchestration Patterns — Implementation of Notebook 21.

This module provides common patterns for orchestrating multiple agents:
1. Router Agent (LLM-based classification)
2. Conditional Router (Rule-based classification)
3. Fan-out / Fan-in (Parallel aggregation)
4. DAG Executor (Directed Acyclic Graph with topological sort)
5. Orchestration Engine (Complex multi-step with triage & review)
6. ASCII Graph Renderer (Visualization)
"""

import json
import re
import time
from typing import List, Dict, Optional, Any, Callable, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass

from config import client, get_model

# ============================================================================
# Base Agent Interface & Wrappers
# ============================================================================

class OrchestratableAgent:
    """Interface for agents that can be used in orchestration patterns."""
    def __init__(self, name: str, specialty: str = "general"):
        self.name = name
        self.specialty = specialty

    def run(self, task: str) -> Dict[str, Any]:
        """Run the agent on a task and return a standardized result."""
        raise NotImplementedError

class SimpleAgent(OrchestratableAgent):
    """A lightweight agent that just calls the LLM with a system prompt."""
    def __init__(self, name: str, role: str, specialty: str = "general"):
        super().__init__(name, specialty)
        self.role = role

    def run(self, task: str) -> Dict[str, Any]:
        t0 = time.time()
        messages = [
            {"role": "system", "content": f"You are {self.name}, a {self.role}. Specialty: {self.specialty}."},
            {"role": "user", "content": task}
        ]
        try:
            response = client.chat.completions.create(
                model=get_model(),
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            content = response.choices[0].message.content or ""
            if not content:
                # Retry once if empty
                response = client.chat.completions.create(
                    model=get_model(),
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content or "No response from model."
        except Exception as e:
            content = f"Error: {e}"
        
        elapsed = time.time() - t0
        return {
            "agent": self.name,
            "specialty": self.specialty,
            "response": content,
            "time": round(elapsed, 2)
        }

# ============================================================================
# Router Agent (LLM-based)
# ============================================================================

class RouterAgent:
    """Classifies incoming tasks and routes to the appropriate specialist."""

    def __init__(self, agents: List[OrchestratableAgent]):
        self.agents = {a.specialty: a for a in agents}
        self.routing_log = []

    def classify(self, task: str) -> str:
        """Use LLM to classify task type."""
        specialties = list(self.agents.keys())
        messages = [
            {"role": "system", "content": f"""You are a task classifier. Classify the task into exactly one category.
Available categories: {specialties}
Reply with ONLY the category name, nothing else."""},
            {"role": "user", "content": task}
        ]
        try:
            response = client.chat.completions.create(
                model=get_model(),
                messages=messages,
                temperature=0.1,
                max_tokens=20,
            )
            result = response.choices[0].message.content or ""
            # Extract matching category
            result_lower = result.strip().lower()
            for s in specialties:
                if s in result_lower:
                    return s
        except Exception:
            pass
        return specialties[0]  # fallback

    def route(self, task: str) -> Dict[str, Any]:
        """Classify and route task to appropriate agent."""
        t0 = time.time()
        category = self.classify(task)
        classify_time = time.time() - t0

        agent = self.agents[category]
        result = agent.run(task)
        result["routed_to"] = category
        result["classify_time"] = round(classify_time, 2)

        self.routing_log.append({
            "task": task[:60],
            "category": category,
            "agent": agent.name
        })
        return result

# ============================================================================
# Conditional Router (Rule-based)
# ============================================================================

class ConditionalRouter:
    """Rule-based routing using keyword patterns and decision trees."""

    def __init__(self, agents: Dict[str, OrchestratableAgent], default_category: str = "general"):
        self.agents = agents
        self.rules: List[Tuple[str, Callable[[str], bool]]] = []
        self.default_category = default_category
        self.route_log = []

    def add_rule(self, category: str, condition: Callable[[str], bool]):
        """Add a routing rule: if condition(task) is True, route to category."""
        self.rules.append((category, condition))

    def classify(self, task: str) -> str:
        task_lower = task.lower()
        for category, condition in self.rules:
            if condition(task_lower):
                return category
        return self.default_category

    def route(self, task: str) -> Dict[str, Any]:
        category = self.classify(task)
        agent = self.agents[category]
        result = agent.run(task)
        result["routed_to"] = category
        result["routing_method"] = "conditional"
        self.route_log.append({"task": task[:50], "category": category})
        return result

# ============================================================================
# Fan-Out / Fan-In (Parallel)
# ============================================================================

class FanOutFanIn:
    """Execute task across multiple agents, then aggregate results."""

    def __init__(self, agents: List[OrchestratableAgent], aggregator: Optional[OrchestratableAgent] = None):
        self.agents = agents
        self.aggregator = aggregator

    def fan_out(self, task: str) -> List[Dict[str, Any]]:
        """Send task to all agents (sequential for now)."""
        results = []
        for agent in self.agents:
            result = agent.run(task)
            results.append(result)
        return results

    def fan_in(self, task: str, results: List[Dict[str, Any]]) -> str:
        """Aggregate results from multiple agents."""
        if self.aggregator is None:
            return self._simple_aggregate(results)

        # Use aggregator agent to synthesize
        combined = "\n".join([
            f"[{r['agent']} ({r['specialty']})]: {r['response']}"
            for r in results
        ])
        agg_task = f"""Original task: {task}

Multiple specialists provided these responses:
{combined}

Synthesize these into a single coherent response. Keep the best insights from each."""
        agg_result = self.aggregator.run(agg_task)
        return agg_result["response"]

    def _simple_aggregate(self, results: List[Dict[str, Any]]) -> str:
        return "\n---\n".join([
            f"**{r['agent']}**: {r['response']}" for r in results
        ])

    def execute(self, task: str) -> Dict[str, Any]:
        t0 = time.time()
        results = self.fan_out(task)
        synthesis = self.fan_in(task, results)
        total_time = time.time() - t0
        return {
            "individual_results": results,
            "synthesis": synthesis,
            "total_time": round(total_time, 2),
            "num_agents": len(self.agents)
        }

# ============================================================================
# DAG Executor
# ============================================================================

class DAGNode:
    """A node in the execution DAG."""
    def __init__(self, node_id: str, agent: OrchestratableAgent, task_template: str):
        self.node_id = node_id
        self.agent = agent
        self.task_template = task_template  # can reference {prev_node_id}
        self.result: Optional[Dict[str, Any]] = None

    def __repr__(self):
        status = "✓" if self.result else "○"
        return f"[{status}] {self.node_id} ({self.agent.name})"


class DAGExecutor:
    """Execute a DAG of agents respecting dependencies."""

    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: List[Tuple[str, str]] = []  # (from, to)

    def add_node(self, node_id: str, agent: OrchestratableAgent, task_template: str):
        self.nodes[node_id] = DAGNode(node_id, agent, task_template)

    def add_edge(self, from_id: str, to_id: str):
        """Add dependency: to_id depends on from_id."""
        self.edges.append((from_id, to_id))

    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological sorting."""
        in_degree = defaultdict(int)
        adjacency = defaultdict(list)

        for node_id in self.nodes:
            in_degree[node_id] = 0

        for src, dst in self.edges:
            adjacency[src].append(dst)
            in_degree[dst] += 1

        queue = [n for n in self.nodes if in_degree[n] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("DAG has a cycle!")
        return order

    def _get_predecessors(self, node_id: str) -> List[str]:
        return [src for src, dst in self.edges if dst == node_id]

    def execute(self, initial_context: str = "") -> Dict[str, Any]:
        """Execute DAG in topological order."""
        order = self._topological_sort()
        results = {}
        t0 = time.time()

        for node_id in order:
            node = self.nodes[node_id]

            # Build task by substituting predecessor results
            task = node.task_template
            if "{context}" in task:
                task = task.replace("{context}", initial_context)
            for pred_id in self._get_predecessors(node_id):
                pred_result = results[pred_id]["response"]
                task = task.replace(f"{{{pred_id}}}", pred_result)

            result = node.agent.run(task)
            results[node_id] = result
            node.result = result

        return {
            "execution_order": order,
            "results": results,
            "total_time": round(time.time() - t0, 2)
        }

# ============================================================================
# Complex Orchestration Engine
# ============================================================================

class OrchestrationEngine:
    """Complex orchestration with triage, parallel dispatch, and quality check."""

    def __init__(self, specialists: Dict[str, OrchestratableAgent], reviewer: OrchestratableAgent):
        self.specialists = specialists
        self.reviewer = reviewer
        self.execution_trace = []

    def triage(self, task: str) -> List[str]:
        """Determine which specialists are needed."""
        messages = [
            {"role": "system", "content": f"""Analyze the task and determine which specialists are needed.
Available: {list(self.specialists.keys())}
Reply with a comma-separated list of needed specialists. Only include those truly relevant."""},
            {"role": "user", "content": task}
        ]
        try:
            response = client.chat.completions.create(
                model=get_model(),
                messages=messages,
                temperature=0.1,
                max_tokens=50,
            )
            result = response.choices[0].message.content or ""
            needed = []
            for spec in self.specialists:
                if spec.lower() in result.lower():
                    needed.append(spec)
            return needed if needed else [list(self.specialists.keys())[0]]
        except Exception:
            return [list(self.specialists.keys())[0]]

    def quality_check(self, task: str, responses: Dict[str, str]) -> Dict[str, Any]:
        """Review quality of responses."""
        combined = "\n".join([f"[{k}]: {v[:200]}" for k, v in responses.items()])
        review_task = f"""Review these specialist responses for quality.
Task: {task}
Responses:
{combined}

Rate overall quality 1-10 and suggest if we need revisions. Reply as: SCORE: X/10 followed by brief assessment."""
        result = self.reviewer.run(review_task)
        # Parse score
        score_match = re.search(r'(\d+)/10', result["response"])
        score = int(score_match.group(1)) if score_match else 5
        return {"score": score, "review": result["response"]}

    def execute(self, task: str) -> Dict[str, Any]:
        self.execution_trace = []
        t0 = time.time()

        # Step 1: Triage
        needed = self.triage(task)
        self.execution_trace.append(f"Triage -> {needed}")

        # Step 2: Parallel specialist work (simulated parallel)
        responses = {}
        for spec_name in needed:
            agent = self.specialists[spec_name]
            result = agent.run(task)
            responses[spec_name] = result["response"]
            self.execution_trace.append(f"{spec_name} -> responded")

        # Step 3: Quality check
        review = self.quality_check(task, responses)
        self.execution_trace.append(f"Review -> {review['score']}/10")

        return {
            "specialists_used": needed,
            "responses": responses,
            "quality_score": review["score"],
            "review": review["review"],
            "total_time": round(time.time() - t0, 2),
            "trace": self.execution_trace
        }

# ============================================================================
# ASCII Graph Renderer
# ============================================================================

class ASCIIGraphRenderer:
    """Render execution graphs as ASCII art."""

    @staticmethod
    def render_dag(nodes: List[str], edges: List[Tuple[str, str]],
                   results: Optional[Dict[str, Any]] = None) -> str:
        # Build adjacency and layers
        in_deg = {n: 0 for n in nodes}
        children = defaultdict(list)
        parents = defaultdict(list)
        for src, dst in edges:
            children[src].append(dst)
            parents[dst].append(src)
            in_deg[dst] += 1

        # Assign layers via BFS
        layers = []
        remaining = set(nodes)
        while remaining:
            layer = [n for n in remaining if all(p not in remaining for p in parents[n])]
            if not layer:
                # Break cycles if any (though DAG should be acyclic)
                node = next(iter(remaining))
                layer = [node]
                remaining.remove(node)
            else:
                remaining -= set(layer)
            layers.append(sorted(layer))

        # Render
        lines = []
        lines.append("+" + "-" * 50 + "+")
        lines.append("|  EXECUTION GRAPH" + " " * 32 + "|")
        lines.append("+" + "-" * 50 + "+")

        for i, layer in enumerate(layers):
            node_strs = []
            for n in layer:
                status = "v" if results and n in results else "o"
                node_strs.append(f"[{status} {n}]")
            layer_str = "  ".join(node_strs)
            padded = f"|  Layer {i}: {layer_str}"
            padded = padded + " " * (51 - len(padded)) + "|"
            lines.append(padded)

            # Draw edges to next layer
            if i < len(layers) - 1:
                next_layer = layers[i + 1]
                edge_parts = []
                for n in layer:
                    for c in children[n]:
                        if c in next_layer:
                            edge_parts.append(f"{n}->{c}")
                if edge_parts:
                    edge_str = ", ".join(edge_parts)
                    edge_line = f"|    v {edge_str}"
                    edge_line = edge_line + " " * (51 - len(edge_line)) + "|"
                    lines.append(edge_line)

        lines.append("+" + "-" * 50 + "+")
        return "\n".join(lines)
