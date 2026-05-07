"""
Mock LLM for testing the agent loop without needing a real API.

This simulates an LLM that "thinks" for a few steps, then provides an answer.
"""

from agent_loop import AgentLoop


def mock_call_llm(messages, **kwargs) -> str:
    """
    A fake LLM that returns predictable responses based on conversation length.
    
    - First call: think step 1
    - Second call: think step 2  
    - Third call: provide the answer
    """
    # Count how many assistant messages we've already had (this is the "turn number")
    assistant_count = sum(1 for msg in messages if msg["role"] == "assistant")
    
    if assistant_count == 0:
        # First turn: initial reasoning
        return '{"action": "think", "thought": "I need to break this problem into parts. Let me start by understanding the core question."}'
    elif assistant_count == 1:
        # Second turn: deeper reasoning
        return '{"action": "think", "thought": "I have the pieces. Now I need to put them together and form a coherent answer."}'
    else:
        # Third turn: confident answer
        return '{"action": "answer", "answer": "The answer is 4. This is because 2+2 equals 4 by basic arithmetic."}'


# Monkey-patch the real call_llm with our mock
import agent_loop
agent_loop.call_llm = mock_call_llm


if __name__ == "__main__":
    # Test with a simple math question
    agent = AgentLoop(
        max_steps=5,
        system_prompt="You are a helpful reasoning agent. Respond with JSON: {\"action\": \"think\", \"thought\": \"...\"} or {\"action\": \"answer\", \"answer\": \"...\"}"
    )
    
    print("Testing with mock LLM...\n")
    state = agent.run("What is 2+2?")
    
    print("\n" + "="*60)
    print("FINAL STATE SUMMARY")
    print("="*60)
    print(state.summary())
    
    print("\n" + "="*60)
    print("CONVERSATION HISTORY")
    print("="*60)
    for i, msg in enumerate(state.messages):
        role = msg["role"].upper()
        content = str(msg["content"])[:80]
        print(f"  [{i}] {role}: {content}...")
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print(f"  Steps taken: {state.current_step}")
    print(f"  Is complete: {state.is_complete}")
    print(f"  Final answer: '{state.final_answer}'")
    print(f"  Mock worked correctly: {state.is_complete and '4' in state.final_answer}")
