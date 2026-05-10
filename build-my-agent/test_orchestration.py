"""
Test suite for Agent Orchestration Patterns.

Verifies:
- RouterAgent (LLM classification)
- ConditionalRouter (Keyword rules)
- FanOutFanIn (Aggregation)
- DAGExecutor (Topological execution)
- OrchestrationEngine (Triage & Review)
"""

import time
from orchestration_patterns import (
    SimpleAgent, RouterAgent, ConditionalRouter, 
    FanOutFanIn, DAGExecutor, OrchestrationEngine,
    ASCIIGraphRenderer
)

def test_orchestration():
    print("Starting Orchestration Patterns Test Suite...")
    
    # Setup Specialist Agents
    code_agent = SimpleAgent("CodeBot", "expert Python programmer", "code")
    math_agent = SimpleAgent("MathBot", "mathematics expert", "math")
    writing_agent = SimpleAgent("WriteBot", "professional writer", "writing")
    research_agent = SimpleAgent("ResearchBot", "research analyst", "research")
    reviewer_agent = SimpleAgent("Reviewer", "quality reviewer", "review")
    
    # 1. Test RouterAgent (LLM)
    print("\n--- Testing RouterAgent (LLM-based) ---")
    router = RouterAgent([code_agent, math_agent, writing_agent, research_agent])
    result = router.route("Write a python function for fibonacci")
    print(f"Task: Write a python function for fibonacci")
    print(f"Routed to: {result['routed_to']} ({result['agent']})")
    assert result['routed_to'] == 'code' or result['agent'] == 'CodeBot'
    
    # 2. Test ConditionalRouter (Rule-based)
    print("\n--- Testing ConditionalRouter (Rule-based) ---")
    cond_router = ConditionalRouter({
        "code": code_agent, "math": math_agent, 
        "writing": writing_agent, "research": research_agent
    })
    cond_router.add_rule("code", lambda t: "python" in t or "code" in t)
    cond_router.add_rule("math", lambda t: "calculate" in t or "sum" in t)
    
    result = cond_router.route("calculate the sum of 1 to 100")
    print(f"Task: calculate the sum of 1 to 100")
    print(f"Routed to: {result['routed_to']} (Method: {result['routing_method']})")
    assert result['routed_to'] == 'math'
    
    # 3. Test FanOutFanIn
    print("\n--- Testing FanOutFanIn ---")
    fan_system = FanOutFanIn(
        agents=[research_agent, writing_agent],
        aggregator=writing_agent
    )
    result = fan_system.execute("The future of AI agents")
    print(f"Task: The future of AI agents")
    print(f"Num agents: {result['num_agents']}")
    for r in result['individual_results']:
        print(f"  Agent {r['agent']} response length: {len(r['response'])}")
    print(f"Synthesis length: {len(result['synthesis'])} chars")
    assert result['num_agents'] == 2
    assert len(result['synthesis']) > 0
    
    # 4. Test DAGExecutor
    print("\n--- Testing DAGExecutor ---")
    dag = DAGExecutor()
    dag.add_node("step1", research_agent, "Research {context}")
    dag.add_node("step2", writing_agent, "Summarize this: {step1}")
    dag.add_edge("step1", "step2")
    
    result = dag.execute("Quantum Computing")
    print(f"Execution order: {result['execution_order']}")
    assert result['execution_order'] == ['step1', 'step2']
    assert 'step1' in result['results']
    assert 'step2' in result['results']
    
    # 5. Test OrchestrationEngine
    print("\n--- Testing OrchestrationEngine ---")
    engine = OrchestrationEngine(
        specialists={"code": code_agent, "math": math_agent},
        reviewer=reviewer_agent
    )
    result = engine.execute("Calculate 123 * 456 and write a python script for it")
    print(f"Specialists used: {result['specialists_used']}")
    print(f"Quality score: {result['quality_score']}/10")
    print(f"Trace: {result['trace']}")
    assert len(result['specialists_used']) > 0
    assert 0 <= result['quality_score'] <= 10
    
    # 6. Test ASCII Renderer
    print("\n--- Testing ASCII Renderer ---")
    renderer = ASCIIGraphRenderer()
    nodes = ["research", "analyze", "report"]
    edges = [("research", "analyze"), ("analyze", "report")]
    art = renderer.render_dag(nodes, edges, results={"research": "", "analyze": ""})
    print(art)
    assert "EXECUTION GRAPH" in art
    
    print("\nAll Orchestration Patterns verified successfully!")

if __name__ == "__main__":
    test_orchestration()
