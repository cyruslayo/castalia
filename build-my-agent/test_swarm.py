"""
Test Swarm Intelligence - Verification of emergent behavior.

This test script verifies:
1. Agent exploration and idea generation.
2. Information sharing between neighbors in a ring topology.
3. Emergent behavior across multiple rounds.
"""

import os
import unittest
from swarm_intelligence import SimpleSwarmAgent, SwarmCoordinator, extract_ideas, rate_idea

class TestSwarmIntelligence(unittest.TestCase):
    
    def test_extract_ideas(self):
        """Test the utility that extracts ideas from LLM text."""
        text = "The first idea is to use blockchain for voting. The second idea is to implement universal basic income using digital currency. The third idea involves using AI to optimize traffic lights."
        ideas = extract_ideas(text)
        self.assertTrue(len(ideas) >= 1, "Should extract at least one idea")
        # Check if any idea contains keywords from the input
        found = any("blockchain" in i.lower() or "voting" in i.lower() or "traffic" in i.lower() for i in ideas)
        self.assertTrue(found, f"None of the extracted ideas {ideas} matched expected keywords")

    def test_roles_initialization(self):
        """Verify agents are initialized with correct roles."""
        coordinator = SwarmCoordinator(n_agents=5, skeptic_ratio=0.4)
        skeptics = [a for a in coordinator.agents if a.role == "skeptic"]
        explorers = [a for a in coordinator.agents if a.role == "explorer"]
        self.assertEqual(len(skeptics), 2)
        self.assertEqual(len(explorers), 3)

    def test_verification_gate(self):
        """Test the verification gate logic."""
        agent = SimpleSwarmAgent(1, role="explorer")
        
        # Good idea should pass (mocked LLM behavior in rate_idea/verify_discovery)
        good_discovery = {"idea": "Using satellite imagery to detect crop disease early.", "source": "agent_2", "gen": 1}
        agent.receive_discovery(good_discovery, threshold=0.1) # Low threshold for test stability
        self.assertTrue(len(agent.received) > 0)
        
        # Bad idea should be throttled (if threshold is high enough)
        agent_strict = SimpleSwarmAgent(2, role="skeptic")
        bad_discovery = {"idea": "Magic beans that grow instantly.", "source": "agent_1", "gen": 1}
        # We can't guarantee LLM score, but we can check if threshold logic works
        agent_strict.receive_discovery(bad_discovery, threshold=1.1) # Impossible threshold
        self.assertEqual(len(agent_strict.received), 0)

    def test_dynamic_topology(self):
        """Verify dynamic neighborhood expansion."""
        coordinator = SwarmCoordinator(n_agents=5, k_neighbors=1)
        # Agent 0 neighbors with k=1 should be 1
        neighbors_normal = coordinator._get_neighbors(0, k_override=1)
        self.assertEqual(len(neighbors_normal), 1)
        
        # Expanded neighbors with k=3
        neighbors_expanded = coordinator._get_neighbors(0, k_override=3)
        self.assertEqual(len(neighbors_expanded), 3)

    def test_swarm_execution_sota(self):
        """Test a full swarm run with SOTA features."""
        topic = "Future of renewable energy"
        coordinator = SwarmCoordinator(n_agents=4, k_neighbors=2)
        results = coordinator.run_swarm(topic, n_rounds=1)
        
        self.assertTrue(results["total_ideas"] > 0)
        # Check if skeptic roles are logged
        agent_reprs = results["agents"]
        self.assertTrue(any("role=skeptic" in r for r in agent_reprs))
        self.assertTrue(any("role=explorer" in r for r in agent_reprs))

    @unittest.skipUnless(
        os.environ.get("RUN_REAL_LLM_TESTS", "").lower() in {"1", "true", "yes"},
        "Set RUN_REAL_LLM_TESTS=1 to exercise the real configured LLM endpoint.",
    )
    def test_swarm_execution_with_real_llm(self):
        """Run the swarm through the real LLM path, not the deterministic fallback."""
        previous = os.environ.get("SWARM_USE_LLM")
        os.environ["SWARM_USE_LLM"] = "1"
        try:
            topic = "Practical uses of AI for disaster response"
            coordinator = SwarmCoordinator(n_agents=2, k_neighbors=1, skeptic_ratio=0.5)
            results = coordinator.run_swarm(topic, n_rounds=1)

            self.assertGreater(results["total_ideas"], 0)
            self.assertGreater(results["unique_ideas"], 0)
            self.assertEqual(len(results["ideas_per_agent"]), 2)
            self.assertTrue(any(agent.ideas for agent in coordinator.agents))
            self.assertTrue(any("role=skeptic" in r for r in results["agents"]))
            self.assertTrue(any("role=explorer" in r for r in results["agents"]))
        finally:
            if previous is None:
                os.environ.pop("SWARM_USE_LLM", None)
            else:
                os.environ["SWARM_USE_LLM"] = previous

if __name__ == "__main__":
    unittest.main()
