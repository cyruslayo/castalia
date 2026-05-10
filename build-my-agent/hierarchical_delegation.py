import time
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from config import get_model, LLM_CONFIG
import openai

# --- Data Structures ---

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REASSIGNED = "reassigned"

@dataclass
class SubTask:
    """A single subtask created by decomposition."""
    id: str
    description: str
    required_skills: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "required_skills": self.required_skills,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "result": self.result,
            "error": self.error,
            "attempts": self.attempts
        }

@dataclass
class TaskDecomposition:
    """Result of decomposing a complex task into subtasks."""
    original_task: str
    subtasks: List[SubTask]
    strategy: str = "flat"  # "flat" or "hierarchical"

    def show(self):
        print(f"\n[TASK] Task: {self.original_task[:80]}...")
        print(f"   Strategy: {self.strategy} | Subtasks: {len(self.subtasks)}")
        for st in self.subtasks:
            status_icon = {
                TaskStatus.PENDING: "[PENDING]",
                TaskStatus.IN_PROGRESS: "[RUNNING]",
                TaskStatus.COMPLETED: "[DONE]",
                TaskStatus.FAILED: "[FAILED]",
                TaskStatus.REASSIGNED: "[RETRY]",
            }.get(st.status, "[?]")
            worker = f" -> {st.assigned_to}" if st.assigned_to else ""
            print(f"   {status_icon} [{st.id}] {st.description[:60]}{worker}")

# --- Worker Agent ---

@dataclass
class WorkerAgent:
    """A specialized worker agent with defined capabilities."""
    name: str
    skills: List[str]
    system_prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7

    def can_handle(self, required_skills: List[str]) -> float:
        """Score how well this worker matches the required skills (0.0 to 1.0)."""
        if not required_skills:
            return 0.5  # Generic match
        matched = sum(1 for s in required_skills if s in self.skills)
        return matched / len(required_skills)

    def execute(self, subtask: SubTask) -> SubTask:
        """Execute a subtask using the LLM."""
        subtask.status = TaskStatus.IN_PROGRESS
        subtask.assigned_to = self.name
        subtask.attempts += 1

        client = openai.OpenAI(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"]
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Task: {subtask.description}"},
        ]

        try:
            response = client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            # Fallback for reasoning models
            if content is None and hasattr(response.choices[0].message, 'reasoning'):
                content = response.choices[0].message.reasoning
                
            if content:
                subtask.result = content
                subtask.status = TaskStatus.COMPLETED
            else:
                subtask.status = TaskStatus.FAILED
                subtask.error = "Empty response from LLM"
        except Exception as e:
            subtask.status = TaskStatus.FAILED
            subtask.error = str(e)

        return subtask

# --- Worker Registry ---

class WorkerRegistry:
    """Registry for discovering workers by capability."""
    def __init__(self):
        self.workers: Dict[str, WorkerAgent] = {}

    def register(self, worker: WorkerAgent):
        self.workers[worker.name] = worker

    def find_worker(self, required_skills: List[str], exclude: List[str] = None) -> Optional[WorkerAgent]:
        exclude = exclude or []
        candidates = [
            (worker, worker.can_handle(required_skills))
            for worker in self.workers.values()
            if worker.name not in exclude
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        if candidates and candidates[0][1] > 0:
            return candidates[0][0]
        return None

# --- Manager Agent ---

class ManagerAgent:
    """Orchestrator that decomposes, delegates, and aggregates."""
    def __init__(self, name: str, registry: WorkerRegistry, max_retries: int = 2):
        self.name = name
        self.registry = registry
        self.max_retries = max_retries
        self.client = openai.OpenAI(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"]
        )

    def _generate(self, messages, max_tokens=1024, temperature=0.4):
        response = self.client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        content = response.choices[0].message.content
        if content is None and hasattr(response.choices[0].message, 'reasoning'):
            content = response.choices[0].message.reasoning
        return content

    def decompose(self, task: str, strategy: str = "flat") -> TaskDecomposition:
        if strategy == "flat":
            prompt = f"""Break the following complex task into 3 distinct, specialized subtasks.
Task: {task}

Available Skills: ["research", "marketing", "writing"]

Respond in this exact JSON format:
{{"subtasks": [
    {{"id": "subtask-1", "description": "Specific action to perform", "skills": ["skill1"]}},
    ...
]}}"""
        else:
            prompt = f"""Break the following complex task into 2-3 high-level phases.
Each phase should contain 2-3 specific subtasks.
Task: {task}

Respond in this exact JSON format:
{{"phases": [
    {{"phase": "phase-name", "subtasks": [
        {{"id": "subtask-id", "description": "What to do", "skills": ["skill1"]}}
    ]}}
]}}"""

        messages = [
            {"role": "system", "content": "You are a project manager. Respond ONLY in valid JSON."},
            {"role": "user", "content": prompt},
        ]

        raw = self._generate(messages)
        subtasks = []
        
        try:
            # Clean up potential markdown blocks
            json_text = raw.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            json_match = re.search(r'\{[\s\S]*\}', json_text)
            if json_match:
                data = json.loads(json_match.group())
                if strategy == "flat":
                    for st in data.get("subtasks", []):
                        subtasks.append(SubTask(id=st["id"], description=st["description"], required_skills=st.get("skills", [])))
                else:
                    for phase in data.get("phases", []):
                        p_name = phase.get("phase", "phase")
                        for st in phase.get("subtasks", []):
                            subtasks.append(SubTask(id=f"{p_name}-{st['id']}", description=st["description"], required_skills=st.get("skills", [])))
        except Exception as e:
            # Fallback
            print(f"Decomposition failed: {e}")
            subtasks = [SubTask("task-1", task, ["research"])]

        return TaskDecomposition(task, subtasks, strategy)

    def delegate(self, decomposition: TaskDecomposition) -> TaskDecomposition:
        for subtask in decomposition.subtasks:
            failed_workers = []
            for attempt in range(self.max_retries):
                worker = self.registry.find_worker(subtask.required_skills, exclude=failed_workers)
                if not worker:
                    subtask.status = TaskStatus.FAILED
                    subtask.error = "No suitable worker found for skills: " + str(subtask.required_skills)
                    break
                
                subtask = worker.execute(subtask)
                if subtask.status == TaskStatus.COMPLETED:
                    break
                else:
                    failed_workers.append(worker.name)
                    if attempt < self.max_retries - 1:
                        subtask.status = TaskStatus.REASSIGNED
        return decomposition

    def aggregate(self, task: str, decomposition: TaskDecomposition) -> str:
        results_text = ""
        for st in decomposition.subtasks:
            status = "COMPLETED" if st.status == TaskStatus.COMPLETED else "FAILED"
            results_text += f"\n--- Subtask: {st.id} ({status}) ---\n"
            results_text += (st.result or st.error or "No output") + "\n"

        prompt = f"""Synthesize the following worker results into a final response for the original task.
Original Task: {task}

Worker Results:
{results_text}

Produce a cohesive, professional report."""
        
        messages = [
            {"role": "system", "content": "You are a senior editor. Synthesize multiple inputs into a cohesive document."},
            {"role": "user", "content": prompt},
        ]
        return self._generate(messages)

    def run(self, task: str, strategy: str = "flat") -> Dict[str, Any]:
        print(f"\n[MANAGER] Manager '{self.name}' starting task...")
        decomp = self.decompose(task, strategy)
        decomp.show()
        
        print(f"\n[SYSTEM] Delegating to workers...")
        decomp = self.delegate(decomp)
        
        print(f"\n[SYSTEM] Aggregating results...")
        final = self.aggregate(task, decomp)
        
        return {
            "final_output": final,
            "decomposition": decomp
        }

if __name__ == "__main__":
    # Quick sanity check
    reg = WorkerRegistry()
    reg.register(WorkerAgent("researcher", ["research"], "Research the topic."))
    reg.register(WorkerAgent("writer", ["writing"], "Write about the topic."))
    
    mgr = ManagerAgent("boss", reg)
    print("Hierarchical Delegation implementation complete (Unicode Safe).")
