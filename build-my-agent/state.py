"""
Agent State - The agent's notebook for tracking everything during execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import time


@dataclass
class AgentState:
    """Complete state of an agent during execution."""
    
    # What are we trying to achieve?
    goal: str = ""
    
    # Conversation history with the LLM (the "whiteboard")
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    # Record of all actions taken (for debugging and tracing)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Which step are we on now?
    current_step: int = 0
    
    # Safety limit to prevent infinite loops
    max_steps: int = 10
    
    # Has the agent finished?
    is_complete: bool = False
    
    # The result to return when done
    final_answer: str = ""
    
    # Diagnostics and timing info
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "start_time": None,
        "end_time": None,
        "total_tokens_estimated": 0,
        "step_times": [],
        "step_token_estimates": [],
    })

    def elapsed_time(self) -> float:
        """Calculate how long the agent has been running."""
        if self.metadata["start_time"] is None:
            return 0.0
        end = self.metadata["end_time"] or time.time()
        return end - self.metadata["start_time"]

    def summary(self) -> str:
        """Return a human-readable summary of the agent's status."""
        status = "COMPLETE" if self.is_complete else "IN PROGRESS"
        answer_preview = self.final_answer[:100]
        if len(self.final_answer) > 100:
            answer_preview += "..."
        
        return (
            f"Agent State [{status}]\n"
            f"  Goal: {self.goal}\n"
            f"  Steps: {self.current_step}/{self.max_steps}\n"
            f"  Time: {self.elapsed_time():.1f}s\n"
            f"  Answer: {answer_preview}"
        )


# Quick test to make sure it works
if __name__ == "__main__":
    state = AgentState(goal="What is 2+2?", max_steps=5)
    print(state.summary())
    print(f"\nFields: {[f.name for f in state.__dataclass_fields__.values()]}")
