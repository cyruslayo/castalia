import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from collections import defaultdict
from config import get_client, get_model

@dataclass
class Message:
    """A structured message between agents."""
    sender: str
    recipient: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"[{self.sender} -> {self.recipient}]: {self.content[:100]}..."

class MessageBus:
    """Central message routing system."""
    def __init__(self):
        self.messages: List[Message] = []
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def send(self, message: Message):
        self.messages.append(message)
        for callback in self.subscribers.get(message.recipient, []):
            callback(message)

    def subscribe(self, agent_name: str, callback: Callable):
        self.subscribers[agent_name].append(callback)

def generate(messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate a response and extract content from <response> tags."""
    client = get_client()
    model = get_model()
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    content = response.choices[0].message.content
    if content is None:
        content = getattr(response.choices[0].message, "reasoning", "")
    
    # 1. Try to extract content between <response> tags
    match = re.search(r"<response>(.*?)</response>", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 2. Try to extract content after </thought>
    if "</thought>" in content.lower():
        parts = re.split(r"</thought>", content, flags=re.IGNORECASE)
        return parts[-1].strip()
        
    # 3. Try to extract content after common reasoning headers
    thinking_headers = [
        "---", 
        "Here's a thinking process:", 
        "Thinking Process:", 
        "Analyze User Input:",
        "**Analyze User Input:**"
    ]
    for header in thinking_headers:
        if header in content:
            # We take the part AFTER the header, but if there are multiple paragraphs,
            # we might need to skip the numbered list that often follows.
            parts = content.split(header)
            potential_content = parts[-1].strip()
            # If it starts with "1.", "2.", etc., it's still thinking.
            lines = potential_content.split("\n")
            for i, line in enumerate(lines):
                # If we find a line that doesn't look like thinking, start from there
                if i > 4 and not line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "**")):
                    return "\n".join(lines[i:]).strip()
            # If we didn't find a clean break, just return the potential content
            content = potential_content

    # 4. Final fallback: Remove leading numbered lists
    lines = content.split("\n")
    if len(lines) > 2 and lines[0].strip().startswith(("1.", "**")):
        for i, line in enumerate(lines):
            if i > 3 and not line.strip().startswith(tuple(str(n)+"." for n in range(1, 10))):
                return "\n".join(lines[i:]).strip()

    return content.strip() or "..."

class ConversableAgent:
    """An agent capable of multi-turn conversation."""
    def __init__(self, name: str, system_prompt: str, bus: MessageBus,
                 max_tokens: int = 500, temperature: float = 0.7):
        self.name = name
        # Even more explicit instructions
        self.system_prompt = system_prompt + "\n\nINSTRUCTION: You MUST use the following format for your response:\n<thought>\n[Your internal reasoning here]\n</thought>\n<response>\n[Your actual message to the other agent here]\n</response>"
        self.bus = bus
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history: List[Message] = [] 
        self.bus.subscribe(self.name, self._on_message)

    def _on_message(self, message: Message):
        self.history.append(message)

    def _build_messages(self) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.history:
            role = "assistant" if msg.sender == self.name else "user"
            # In the history, we only store the "response" part to keep it clean for the next turn
            messages.append({"role": role, "content": msg.content})
        return messages

    def respond(self, to: str) -> Message:
        messages = self._build_messages()
        response_text = generate(messages, max_tokens=self.max_tokens, temperature=self.temperature)
        # We only save the extracted response to history to avoid re-parsing monologues in future turns
        msg = Message(sender=self.name, recipient=to, content=response_text)
        self.history.append(msg)
        self.bus.send(msg)
        return msg

class TwoAgentChat:
    """Orchestrator for a dialogue between two agents."""
    def __init__(self, agent_a, agent_b, bus, max_turns=10, termination_keyword="<DONE>"):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.bus = bus
        self.max_turns = max_turns
        self.termination_keyword = termination_keyword
        self.turns = 0

    def start(self, initial_message: str):
        print(f"\n{'='*50}\nCONVERSATION: {self.agent_a.name} <-> {self.agent_b.name}\n{'='*50}\n")
        initial_msg = Message(sender="User", recipient=self.agent_a.name, content=initial_message)
        self.agent_a._on_message(initial_msg)
        
        current_agent = self.agent_a
        next_agent = self.agent_b

        while self.turns < self.max_turns:
            msg = current_agent.respond(to=next_agent.name)
            print(f"[{current_agent.name}]: {msg.content}\n")
            
            if self.termination_keyword.lower() in msg.content.lower():
                print(f"--- Termination signal detected: {self.termination_keyword} ---")
                break
                
            current_agent, next_agent = next_agent, current_agent
            self.turns += 1
            
        if self.turns >= self.max_turns:
            print(f"--- Limit reached ({self.max_turns} turns) ---")
        print(f"\n{'='*50}\n")

def run_experiment(name_a, prompt_a, name_b, prompt_b, initial_msg, max_turns=6):
    bus = MessageBus()
    agent_a = ConversableAgent(name_a, prompt_a.replace("[TERMINATE]", "<DONE>"), bus)
    agent_b = ConversableAgent(name_b, prompt_b.replace("[TERMINATE]", "<DONE>"), bus)
    chat = TwoAgentChat(agent_a, agent_b, bus, max_turns=max_turns, termination_keyword="<DONE>")
    chat.start(initial_msg)

if __name__ == "__main__":
    # Experiment 1: Researcher + Skeptic
    print("Running Experiment 1: Researcher + Skeptic")
    researcher_prompt = "You are a bold Scientific Researcher. Propose innovative but slightly unproven theories. Keep responses concise. If you agree with the feedback, say <DONE>."
    skeptic_prompt = "You are a harsh Skeptic. Find flaws in logic and demand data. Be brief and critical. If you have no more critiques, say <DONE>."
    run_experiment("Researcher", researcher_prompt, "Skeptic", skeptic_prompt, "Tell me your theory on why cats land on their feet using quantum entanglement.")

    # Experiment 2: Teacher + Student
    print("\nRunning Experiment 2: Teacher + Student")
    teacher_prompt = "You are a patient Teacher. Explain complex concepts using simple analogies. Ask the student if they understand. If they do, say <DONE>."
    student_prompt = "You are a curious Student. Ask follow-up questions and summarize what you learned. If you understand perfectly, say <DONE>."
    run_experiment("Teacher", teacher_prompt, "Student", student_prompt, "Explain how a transformer model works to a 10-year old.")

    # Experiment 3: Planner + Critic
    print("\nRunning Experiment 3: Planner + Critic")
    planner_prompt = "You are a Strategic Planner. Create detailed project plans with milestones. If the critic approves, say <DONE>."
    critic_prompt = "You are a Project Critic. Identify risks, missing dependencies, and unrealistic timelines. If the plan is solid, say <DONE>."
    run_experiment("Planner", planner_prompt, "Critic", critic_prompt, "Create a plan to colonize Mars in the next 10 years.")
