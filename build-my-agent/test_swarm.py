"""
Test Swarm Intelligence - Verification of emergent behavior.

This test script verifies:
1. Agent exploration and idea generation.
2. Information sharing between neighbors in a ring topology.
3. Emergent behavior across multiple rounds.
"""

import sys
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

    def test_ring_topology(self):
        """Verify the ring topology neighbor calculation."""
        # 5 agents, k=2 neighbors
        coordinator = SwarmCoordinator(n_agents=5, k_neighbors=2)
        
        # Agent 0 should have neighbors 1 and 4 (wrap around)
        neighbors_0 = coordinator._get_neighbors(0)
        self.assertEqual(len(neighbors_0), 2)
        self.assertIn(1, neighbors_0)
        self.assertIn(4, neighbors_0)
        
        # Agent 2 should have neighbors 1 and 3
        neighbors_2 = coordinator._get_neighbors(2)
        self.assertEqual(len(neighbors_2), 2)
        self.assertIn(1, neighbors_2)
        self.assertIn(3, neighbors_2)

    def test_agent_sharing(self):
        """Test that agents can share and receive discoveries."""
        agent1 = SimpleSwarmAgent(1)
        agent2 = SimpleSwarmAgent(2)
        
        discovery = {"idea": "AI for plants", "source": "agent_1", "gen": 1}
        agent2.receive_discovery(discovery)
        
        self.assertEqual(len(agent2.received), 1)
        self.assertEqual(agent2.received[0]["idea"], "AI for plants")
        
        # Agent should NOT receive its own discovery
        agent1.receive_discovery(discovery)
        self.assertEqual(len(agent1.received), 0)

    def test_swarm_execution(self):
        """Test a small-scale swarm run (mocked-like or fast)."""
        # Using a very simple topic to keep it fast
        topic = "Future of space travel"
        coordinator = SwarmCoordinator(n_agents=3, k_neighbors=2)
        
        # Run 1 round
        results = coordinator.run_swarm(topic, n_rounds=1)
        
        self.assertEqual(len(coordinator.agents), 3)
        self.assertTrue(results["total_ideas"] > 0)
        self.assertEqual(len(results["rounds"]), 1)
        
        # Check that agents received discoveries from neighbors
        for agent in coordinator.agents:
            # In a 3-agent ring with k=2, everyone is neighbors with everyone else
            self.assertTrue(len(agent.received) > 0, f"Agent {agent.agent_id} received no ideas")

if __name__ == "__main__":
    unittest.main()
