from hierarchical_delegation import WorkerAgent, WorkerRegistry, ManagerAgent
import json

def run_live_test():
    # 1. Setup Registry with specialized workers
    reg = WorkerRegistry()
    
    reg.register(WorkerAgent(
        name="researcher", 
        skills=["research"], 
        system_prompt="You are a meticulous market researcher. Identify 3 unique selling points (USPs) for the product and explain why they appeal to tech-savvy consumers."
    ))
    
    reg.register(WorkerAgent(
        name="marketer", 
        skills=["marketing"], 
        system_prompt="You are a creative brand strategist. Based on product features, define a compelling 'hook' and identify the primary target audience."
    ))
    
    reg.register(WorkerAgent(
        name="writer", 
        skills=["writing"], 
        system_prompt="You are a professional copywriter. Write a 2-paragraph promotional blast that is punchy, persuasive, and highlights the caffeine tracking feature."
    ))

    # 2. Setup Manager
    mgr = ManagerAgent("AgencyDirector", reg)

    # 3. Define the complex task
    task = "Create a promotional campaign for 'NeoCup', an AI-powered coffee mug that maintains temperature and tracks caffeine intake via a mobile app."
    
    print(f"--- STARTING HIERARCHICAL DELEGATION TEST ---")
    print(f"Task: {task}\n")

    # 4. Run the full DDA cycle
    result = mgr.run(task, strategy="flat")

    # 5. Output Verification
    print("\n" + "="*60)
    print("FINAL SYNTHESIZED REPORT")
    print("="*60)
    print(result["final_output"])
    print("="*60)

    # Log some stats
    decomp = result["decomposition"]
    print(f"\nStats:")
    print(f"- Subtasks Created: {len(decomp.subtasks)}")
    for st in decomp.subtasks:
        print(f"  * [{st.id}] Assigned to: {st.assigned_to} | Status: {st.status.value}")

if __name__ == "__main__":
    run_live_test()
