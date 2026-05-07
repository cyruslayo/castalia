"""
Notebook 08: Tree of Thought (ToT)
=====================================

The core idea: instead of following one reasoning path (like ReAct or planning),
generate MULTIPLE candidate next steps, evaluate each one, and follow only
the most promising. Like a chess player considering several moves ahead.

Three key operations at each step:
1. GENERATE: Create N different possible next thoughts
2. EVALUATE: Score each thought (0-10) on how promising it is
3. PRUNE: Keep only top-K thoughts above a threshold, discard the rest

Two search strategies:
- BFS (Breadth-First Search): Explore all thoughts at depth 1, then all at depth 2
- DFS (Depth-First Search): Go deep on one branch, then backtrack

By the end of this module, you will have:
- ThoughtNode: The tree data structure
- TreeOfThought: The engine with generate/evaluate/BFS/DFS
- A working test with real LLM calls
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import List, Optional

from config import get_client, get_model


# ====================================================================
# Part 1: The ThoughtNode - One Step in the Reasoning Tree
# ====================================================================

@dataclass
class ThoughtNode:
    """
    A single node in the thought tree.

    Each node represents one reasoning step. A node has:
    - content: The actual thought/reasoning step
    - score: How promising this step is (0-10), assigned by the evaluator
    - children: The next possible thoughts (generated from this one)
    - parent: Where this thought came from (for path reconstruction)
    - depth: How many steps from the root
    - node_id: Unique identifier for debugging/visualization
    - is_terminal: True if this represents a complete solution
    """
    content: str
    score: float = 0.0
    children: List['ThoughtNode'] = field(default_factory=list)
    parent: Optional['ThoughtNode'] = None
    depth: int = 0
    node_id: int = 0
    is_terminal: bool = False

    def add_child(self, content: str, score: float = 0.0) -> 'ThoughtNode':
        """
        Create and attach a child thought.

        This is how the tree grows. When we generate candidate next steps,
        each one becomes a child of the current node.
        """
        child = ThoughtNode(
            content=content,
            score=score,
            parent=self,
            depth=self.depth + 1,
        )
        self.children.append(child)
        return child

    def get_path(self) -> List['ThoughtNode']:
        """
        Reconstruct the path from root to this node.

        Walk up the parent chain, then reverse to get root-first order.
        This is the "reasoning chain" that leads to this thought.
        """
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def path_text(self, max_len: int = 60) -> str:
        """
        Get the full reasoning chain as a readable string.

        Format: "Root thought -> Step 1 thought -> This thought"
        Truncated to max_len chars per step for readability.
        """
        return " -> ".join(n.content[:max_len] for n in self.get_path())

    def __str__(self) -> str:
        return "ThoughtNode(id={}, depth={}, score={:.1f}, content='{}...')".format(
            self.node_id, self.depth, self.score, self.content[:40]
        )


# ====================================================================
# Part 2: The TreeOfThought Engine
# ====================================================================

class TreeOfThought:
    """
    The Tree of Thought reasoning engine.

    This class manages the full ToT cycle:
    1. Generate multiple candidate next steps from a given node
    2. Evaluate each candidate (score 0-10)
    3. Prune: keep only the top-K candidates above a threshold
    4. Repeat for max_depth levels

    Two search strategies are supported:
    - BFS (breadth-first): Explore all candidates at one depth before going deeper
    - DFS (depth-first): Follow one promising branch all the way, then backtrack

    Configurable parameters:
    - branching_factor: How many candidate thoughts to generate per node (default: 3)
    - max_depth: How many levels deep to explore (default: 3)
    - score_threshold: Minimum score to keep a thought (below this = pruned, default: 3.0)
    - top_k: From all viable children, keep only the top K (default: 2)
    """

    def __init__(
        self,
        branching_factor: int = 3,
        max_depth: int = 3,
        score_threshold: float = 3.0,
        top_k: int = 2,
    ):
        # How many candidate thoughts to generate per node
        self.branching_factor = branching_factor
        # How deep the tree can grow
        self.max_depth = max_depth
        # Prune any thought scoring below this
        self.score_threshold = score_threshold
        # From viable thoughts, keep only the top K
        self.top_k = top_k

        # Internal state
        self.node_counter = 0
        self.total_llm_calls = 0
        self.all_nodes: List[ThoughtNode] = []

    def _next_id(self) -> int:
        """Generate a unique node ID."""
        self.node_counter += 1
        return self.node_counter

    # ------------------------------------------------------------------
    # Part 2a: Generate - Create candidate next thoughts
    # ------------------------------------------------------------------

    def generate_thoughts(
        self,
        node: ThoughtNode,
        problem: str,
        n: Optional[int] = None,
    ) -> List[ThoughtNode]:
        """
        Generate N candidate next thoughts from the current node.

        The LLM is asked to produce diverse, distinct next steps.
        Each candidate explores a different approach or continuation.

        Args:
            node: The current thought we're branching from
            problem: The original problem statement
            n: How many candidates to generate (defaults to branching_factor)

        Returns:
            List of new ThoughtNode children added to the parent node
        """
        n = n or self.branching_factor
        path_text = node.path_text()

        # Build the prompt: ask for diverse, numbered candidate thoughts
        prompt = ("You are a problem solver exploring a mathematical puzzle. You need to generate concrete, specific next steps.\n\n"
        "Given the problem and current reasoning, generate EXACTLY {} DIFFERENT possible next steps.\n"
        "Each step must be a concrete ACTION - a specific calculation, strategy, or observation.\n"
        "DO NOT generate abstract meta-commentary like 'analyze the problem' or 'evaluate'.\n\n"
        "Problem: {}\n\n"
        "Current reasoning: {}\n\n"
        "Now generate {} different next steps. Each should be a numbered item with a SPECIFIC mathematical approach or calculation.\n\n"
        "Example format:\n"
        "1. First, arrange the digits in descending order to get the maximum number: 9876543210. Then check...\n"
        "2. Remember that divisibility by 12 requires both divisibility by 3 and by 4. Start by...\n"
        "3. Consider the constraint that the last two digits determine divisibility by 4. Try...").format(n, problem, path_text, n)

        client = get_client()
        response = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,  # Higher for reasoning + content
            temperature=0.8,  # Higher temperature for diverse candidates
        )
        self.total_llm_calls += 1

        raw = response.choices[0].message.content
        if raw is None or raw.strip() == "":
            # Fallback: this model puts output in reasoning field first
            raw = response.choices[0].message.reasoning or ""

        if raw is None:
            raw = ""

        # Parse the numbered list
        thought_texts = self._parse_numbered_list(raw, n)

        # Create child nodes for each thought
        children = []
        for text in thought_texts:
            child = node.add_child(text)
            child.node_id = self._next_id()
            self.all_nodes.append(child)
            children.append(child)

        return children

    def _parse_numbered_list(self, text: str, expected_n: int) -> List[str]:
        """
        Parse a numbered list from the LLM response.

        Primary: Match "1. content", "2. content" pattern
        Fallback: Split by sentences if we get fewer than expected
        """
        thoughts = []

        # Primary: extract numbered items
        for line in text.split("\n"):
            line = line.strip()
            match = re.match(r"^\d+[\.)]\s+(.+)", line)
            if match and len(match.group(1)) > 10:
                thoughts.append(match.group(1).strip())

        # Fallback: if we got fewer than expected, try sentence splitting
        if len(thoughts) < expected_n:
            sentences = [
                s.strip()
                for s in text.split(".")
                if len(s.strip()) > 15
            ]
            while len(thoughts) < expected_n and sentences:
                thoughts.append(sentences.pop(0))

        return thoughts

    # ------------------------------------------------------------------
    # Part 2b: Evaluate - Score how promising a thought is
    # ------------------------------------------------------------------

    def evaluate_thought(
        self,
        node: ThoughtNode,
        problem: str,
    ) -> float:
        """
        Evaluate how promising a single thought is on a 0-10 scale.

        The LLM acts as a "judge" that assesses whether this reasoning
        step is likely to lead to a solution.

        Scoring rubric:
        - 0-3: Dead end, incorrect, or unhelpful
        - 4-6: Partially correct, might lead somewhere
        - 7-9: Strong progress, likely leads to solution
        - 10: Correct and complete solution

        Args:
            node: The thought to evaluate
            problem: The original problem

        Returns:
            The assigned score (0-10)
        """
        path_text = node.path_text()

        # Build prompt using safe string concatenation
        intro = (
            "You are evaluating a reasoning step for solving a problem.\n\n"
            "Rate how promising this reasoning path is on a scale of 0-10:\n"
            "- 0-3: Dead end, incorrect, or unhelpful\n"
            "- 4-6: Partially correct, might lead somewhere\n"
            "- 7-9: Strong progress, likely leads to solution\n"
            "- 10: Correct and complete solution\n\n"
        )
        middle = "Problem: {}\n\nReasoning path: {}\n\n".format(problem, path_text)
        ending = "Respond with ONLY a JSON object with 'score' and 'reason' fields."
        prompt = intro + middle + ending

        client = get_client()
        response = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,  # Higher for reasoning + content
            temperature=0.3,  # Deterministic evaluation
        )
        self.total_llm_calls += 1

        raw = response.choices[0].message.content
        if raw is None or raw.strip() == "":
            # Fallback: this model puts output in reasoning field first
            raw = response.choices[0].message.reasoning or ""

        if raw is None:
            raw = ""

        # Try to parse the score from JSON
        score = self._extract_score(raw)

        # Clamp to 0-10
        node.score = min(10.0, max(0.0, score))
        return node.score

    def _extract_score(self, text: str) -> float:
        """
        Extract a numeric score from the LLM response.

        Strategy 1: Find JSON object, extract "score" field
        Strategy 2: Find first number in the text
        Strategy 3: Return default of 5.0
        """
        # Strategy 1: JSON parsing
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = float(data.get("score", 5.0))
                return score
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Strategy 2: Find first number
        num_match = re.search(r"(\d+(?:\.\d+)?)", text)
        if num_match:
            return float(num_match.group(1))

        # Strategy 3: Default
        return 5.0

    # ------------------------------------------------------------------
    # Part 2c: BFS Search - Explore level by level
    # ------------------------------------------------------------------

    def bfs_search(
        self,
        problem: str,
        root: Optional[ThoughtNode] = None,
    ) -> ThoughtNode:
        """
        Breadth-First Search through the thought tree.

        At each depth level:
        1. For every node at this level, generate N candidate children
        2. Evaluate each child
        3. Prune: keep only children with score >= threshold
        4. From viable children, keep only top-K
        5. Move to next depth with the survivors

        This explores WIDELY first, ensuring we don't miss a good branch
        that starts with a slightly lower first-step score.

        Args:
            problem: The problem to solve
            root: The starting node (created from the problem if None)

        Returns:
            The best node found (highest score, or a terminal solution node)
        """
        if root is None:
            root = ThoughtNode(
                content="Problem: " + problem,
                node_id=0,
            )
            self.all_nodes = [root]

        current_level = [root]
        best_node = root

        for depth in range(self.max_depth):
            print("\n[BFS Depth {}] Expanding {} nodes...".format(depth + 1, len(current_level)))

            next_level = []

            for node in current_level:
                # Generate candidate next thoughts
                children = self.generate_thoughts(node, problem)

                for child in children:
                    # Evaluate each candidate
                    score = self.evaluate_thought(child, problem)
                    print("  Node {}: score={:.1f} - {}".format(
                        child.node_id,
                        score,
                        child.content[:70]
                    ))

                    # Track the overall best
                    if score > best_node.score:
                        best_node = child

                    # Early termination: if we found a near-perfect solution
                    if child.score >= 9.0:
                        child.is_terminal = True
                        print("  ** Solution found at depth {}!".format(depth + 1))
                        return best_node

                # Pruning: keep only viable children (above threshold)
                viable = [c for c in children if c.score >= self.score_threshold]
                viable.sort(key=lambda x: x.score, reverse=True)
                next_level.extend(viable[:self.top_k])

            # Move to next level
            current_level = next_level

            if not current_level:
                print("  No viable nodes remaining - search exhausted.")
                break

        return best_node

    # ------------------------------------------------------------------
    # Part 2d: DFS Search - Go deep, then backtrack
    # ------------------------------------------------------------------

    def dfs_search(
        self,
        problem: str,
        root: Optional[ThoughtNode] = None,
    ) -> ThoughtNode:
        """
        Depth-First Search through the thought tree.

        At each node:
        1. Generate candidate children
        2. For each child (in order of generation):
           a. Evaluate it
           b. If score >= 9.0: found a solution, return immediately
           c. If score >= threshold: recurse into this child
           d. Otherwise: skip this branch (prune)

        DFS goes DEEP first. This can be faster when a good solution
        is down a particular branch, but risks missing better branches
        that start with slightly lower scores.

        Args:
            problem: The problem to solve
            root: The starting node

        Returns:
            The best node found
        """
        if root is None:
            root = ThoughtNode(
                content="Problem: " + problem,
                node_id=0,
            )
            self.all_nodes = [root]

        best_node = root

        def _dfs(inner_node: ThoughtNode, depth: int):
            nonlocal best_node

            if depth >= self.max_depth:
                return

            # Generate candidates
            children = self.generate_thoughts(inner_node, problem)

            for child in children:
                score = self.evaluate_thought(child, problem)
                indent = "  " * (depth + 1)
                print("{}Node {}: score={:.1f} - {}".format(
                    indent,
                    child.node_id,
                    score,
                    child.content[:60]
                ))

                # Track overall best
                if score > best_node.score:
                    best_node = child

                # Early termination for excellent solutions
                if score >= 9.0:
                    child.is_terminal = True
                    return

                # Only recurse into promising branches
                if score >= self.score_threshold:
                    _dfs(child, depth + 1)

        _dfs(root, 0)
        return best_node

    # ------------------------------------------------------------------
    # Part 2e: Tree Visualization
    # ------------------------------------------------------------------

    def get_best_path(self) -> List[ThoughtNode]:
        """
        Find the highest-scoring leaf node and return its path.

        A leaf is a node with no children. If all nodes have children,
        fall back to the highest-scoring node overall.
        """
        if not self.all_nodes:
            return []

        # Find leaf nodes (no children)
        leaves = [n for n in self.all_nodes if not n.children]
        if not leaves:
            leaves = self.all_nodes

        best_leaf = max(leaves, key=lambda n: n.score)
        return best_leaf.get_path()

    def visualize_tree(
        self,
        root: Optional[ThoughtNode] = None,
        max_content_len: int = 50,
    ) -> str:
        """
        Generate an ASCII visualization of the thought tree.

        Format:
        Root (score: 0.0)
        +-- Child A (score: 7.0)
        |   +-- A1 (score: 8.5)
        |   +-- A2 (score: 3.0)
        +-- Child B (score: 5.0)

        Args:
            root: The root of the tree to visualize
            max_content_len: Maximum characters for the content preview

        Returns:
            A string containing the tree visualization
        """
        if root is None and self.all_nodes:
            root = self.all_nodes[0]
        if root is None:
            return "(no tree to visualize)"

        lines = []

        def _render(node: ThoughtNode, prefix: str, is_last: bool):
            connector = "+-- " if is_last else "+-- "
            content_preview = node.content[:max_content_len]
            if len(node.content) > max_content_len:
                content_preview += "..."

            solution_tag = " (SOLUTION)" if node.is_terminal else ""
            line = "{}{}{} (score: {:.1f}){}".format(
                prefix,
                connector if node.depth > 0 else "",
                content_preview,
                node.score,
                solution_tag
            )
            lines.append(line)

            # Calculate the new prefix for children
            new_prefix = prefix + ("    " if is_last else "    ")

            if node.children:
                for i, child in enumerate(node.children):
                    is_last_child = (i == len(node.children) - 1)
                    _render(child, new_prefix, is_last_child)

        _render(root, "", True)
        return "\n".join(lines)

    def summary(self, best_node: Optional[ThoughtNode] = None) -> dict:
        """
        Return a summary of the search.
        """
        if best_node is None:
            best_node = self.all_nodes[0] if self.all_nodes else None

        best_path = best_node.get_path() if best_node else []

        return {
            "total_nodes": len(self.all_nodes),
            "total_llm_calls": self.total_llm_calls,
            "best_score": best_node.score if best_node else 0,
            "best_depth": best_node.depth if best_node else 0,
            "best_path_length": len(best_path),
            "is_terminal": best_node.is_terminal if best_node else False,
            "best_path": [
                {"depth": n.depth, "score": n.score, "content": n.content[:100]}
                for n in best_path
            ],
        }


# ====================================================================
# Part 3: Fast Test (No LLM calls) - Validate the data structures
# ====================================================================

def _test_thought_node():
    """Test the ThoughtNode data structure without any LLM calls."""
    print("=" * 60)
    print("Fast Test 1: ThoughtNode Structure")
    print("=" * 60)

    # Create a simple tree
    root = ThoughtNode(content="Problem: Make 24 from 4, 7, 8, 3", node_id=0)

    # Add children (candidate first steps)
    a = root.add_child("Try 8 * 3 = 24, then adjust with 4 and 7", score=7.0)
    b = root.add_child("Try 7 * 4 = 28, need to subtract 4", score=5.0)
    c = root.add_child("Try (4+8)*3 - 7 = 29, too high", score=3.0)

    # Add grandchildren (next steps from A)
    a1 = a.add_child("8 * 3 = 24, but we need 4 and 7: (7-4)+8*3 = 27, no", score=3.0)
    a2 = a.add_child("Try 3*8+7-4=27, close. Try other combos.", score=6.0)

    print("  Root: {}".format(root.content))
    print("  Children of root: {}".format(len(root.children)))
    print("  A: score={}, depth={}".format(a.score, a.depth))
    print("  A1 path: {}".format(a1.path_text()))
    print("  A2 path: {}".format(a2.path_text()))
    print("  C (low score, would be pruned): score={}".format(c.score))

    # Verify
    assert len(root.children) == 3, "Root should have 3 children"
    assert a.depth == 1, "A should be at depth 1"
    assert a1.depth == 2, "A1 should be at depth 2"
    assert a1.parent is a, "A1's parent should be A"
    assert len(a1.get_path()) == 3, "Path from root to A1 should be 3 nodes"

    print("  Fast Test 1 PASSED\n")
    return root, a, b, c, a1, a2


def _test_visualization():
    """Test the tree visualization without LLM calls."""
    print("=" * 60)
    print("Fast Test 2: Tree Visualization")
    print("=" * 60)

    root = ThoughtNode(content="Problem: Make 24 from 4, 7, 8, 3", node_id=0)
    a = root.add_child("Try 8 * 3 = 24, then adjust with 4 and 7", score=7.0)
    b = root.add_child("Try 7 * 4 = 28, need to subtract 4", score=5.0)
    c = root.add_child("Try (4+8)*3 - 7 = 29, too high", score=3.0)
    a1 = a.add_child("8*3=24 but 4,7 unused: (7-4)+24=27, no", score=3.0)
    a2 = a.add_child("Try 3*8+7-4=27, close. Try other combos.", score=6.0)
    a1.is_terminal = True  # Mark as solution for testing

    vis = root.path_text()  # Simple path text
    print("  Path to A1: " + a1.path_text())
    print("  A1 is terminal: {}".format(a1.is_terminal))

    # Test the visualization function
    engine = TreeOfThought()
    engine.all_nodes = [root, a, b, c, a1, a2]

    tree_str = engine.visualize_tree(root, max_content_len=40)
    print("  Tree visualization:")
    for line in tree_str.split("\n"):
        print("    " + line)

    assert "SOLUTION" in tree_str, "Should mark terminal node"

    print("  Fast Test 2 PASSED\n")


def _test_pruning_logic():
    """Test that pruning correctly keeps only viable candidates."""
    print("=" * 60)
    print("Fast Test 3: Pruning Logic")
    print("=" * 60)

    root = ThoughtNode(content="Test problem", node_id=0)

    # Add children with various scores
    scores = [9.0, 8.0, 7.5, 6.0, 4.0, 3.5, 2.0, 1.0]
    for s in scores:
        root.add_child("Candidate with score {}".format(s), score=s)

    all_children = root.children

    # Simulate the pruning logic from BFS
    threshold = 3.0
    top_k = 2

    viable = [c for c in all_children if c.score >= threshold]
    viable.sort(key=lambda x: x.score, reverse=True)
    kept = viable[:top_k]

    print("  Total children: {}".format(len(all_children)))
    print("  Above threshold ({}): {}".format(threshold, len(viable)))
    print("  Kept (top {}): scores = {}".format(top_k, [c.score for c in kept]))

    assert len(viable) == 6, "6 children should be above threshold 3.0 (scores 9,8,7.5,6,4,3.5)"
    assert len(kept) == 2, "Should keep only top 2"
    assert kept[0].score == 9.0, "Top kept should be 9.0"
    assert kept[1].score == 8.0, "Second kept should be 8.0"

    print("  Fast Test 3 PASSED: Pruning correctly keeps top candidates\n")


# ====================================================================
# Part 4: LLM Test - Run a small ToT search
# ====================================================================

def test_tot_with_llm(search="bfs"):
    """
    Run a small Tree of Thought with real LLM calls.

    Uses a simple puzzle that benefits from exploring multiple approaches.
    """
    print("=" * 60)
    print("LLM Test: Tree of Thought ({} Search, 2 levels)".format(search.upper()))
    print("=" * 60)

    problem = (
        "You have the digits 4, 1, 5, 9, 0, 6, 8, 3, 7, 2 (each used exactly once). "
        "Form the largest possible 10-digit number that is divisible by 12. "
        "Show your reasoning."
    )

    print("\n  Problem: {}...".format(problem[:80]))
    print("  Search: {} (generate 3 candidates, keep top 2 above score 3.0, max 2 levels deep)\n".format(search.upper()))

    # Configure: small search to keep it fast
    tot = TreeOfThought(
        branching_factor=3,
        max_depth=2,  # Only 2 levels for a fast test
        score_threshold=3.0,
        top_k=2,
    )

    root = ThoughtNode(content="Problem: " + problem, node_id=0)
    tot.all_nodes = [root]

    # Run the search
    if search == "bfs":
        best = tot.bfs_search(problem, root=root)
    else:
        best = tot.dfs_search(problem, root=root)

    # Show the results
    summary = tot.summary(best)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("  Total nodes created: {}".format(summary["total_nodes"]))
    print("  Total LLM calls: {}".format(summary["total_llm_calls"]))
    print("  Best score: {:.1f}/10".format(summary["best_score"]))
    print("  Best depth: {}".format(summary["best_depth"]))
    print("  Is terminal (solution found): {}".format(summary["is_terminal"]))

    if summary["best_path"]:
        print("\n  Best reasoning path:")
        for entry in summary["best_path"]:
            safe_content = entry["content"][:80].encode('ascii', errors='replace').decode('ascii')
            print("    [Depth {}] (score: {:.1f}) {}".format(
                entry["depth"], entry["score"], safe_content
            ))

    # Show tree visualization
    print("\n  Tree visualization:")
    vis = tot.visualize_tree(root, max_content_len=45)
    for line in vis.split("\n"):
        print("    " + line)

    print("\n  LLM Test: {} Search completed".format(search.upper()))
    return summary


# ====================================================================
# Main entry point
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Tree of Thought - Module 08")
    print("=" * 60)

    # Part 1: Fast tests (no LLM)
    _test_thought_node()
    _test_visualization()
    _test_pruning_logic()

    # Part 2: LLM test
    import sys
    # Default to BFS, but allow "dfs" as argument
    search_type = "bfs"
    if len(sys.argv) > 1:
        search_type = sys.argv[1].lower()

    if search_type in ("bfs", "dfs"):
        test_tot_with_llm(search=search_type)
    else:
        print("Usage: python tree_of_thought.py [bfs|dfs]")

    print("\nModule 08 complete.")
