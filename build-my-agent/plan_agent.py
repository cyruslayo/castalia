"""
Notebook 06: Plan and Execute
===============================

The core idea: instead of deciding one step at a time (reactive ReAct),
first create a complete plan, then execute it step by step (proactive planning).

This is the difference between improvising a speech vs writing an outline first.

Architecture:
1. Planner: Decomposes the goal into an ordered list of sub-tasks
2. Executor: Runs each sub-task using ReAct-style tool use
3. Coordinator: Combines them, handles re-planning when steps fail
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

# Reuse our existing modules
from react_agent import ReActAgent
from tools import generate_tools_instruction
from guard import guarded_execute_tool
from parser import parse_response
from config import get_client, get_model


# ====================================================================
# Part 1: The Plan Data Structure
# ====================================================================

@dataclass
class PlanStep:
    """One step in the execution plan."""
    number: int
    description: str
    dependencies: list = field(default_factory=list)


@dataclass
class Plan:
    """A complete plan: a list of ordered, potentially dependent steps."""
    summary: str
    steps: list = field(default_factory=list)

    def total_steps(self) -> int:
        return len(self.steps)


# ====================================================================
# Part 2: The Planner Prompt
# ====================================================================

def build_planner_prompt(goal: str, tools_instruction: str) -> str:
    return (
        "You are a planning assistant. Your job is to break down a complex goal into an ordered plan.\n"
        "\n"
        "## The Goal\n"
        + goal + "\n"
        "\n"
        "## Available Tools\n"
        + tools_instruction + "\n"
        "\n"
        "## Your Task\n"
        "Create a plan that breaks the goal into clear, actionable steps. Each step should be:\n"
        "- Specific enough to execute independently\n"
        "- Ordered in a logical sequence\n"
        "- Labeled with any dependencies on previous steps\n"
        "\n"
        "## Output Format\n"
        "Return a SINGLE JSON object with this exact structure:\n"
        '{\n'
        '  "summary": "One sentence summarizing the overall plan.",\n'
        '  "steps": [\n'
        '    {\n'
        '      "number": 1,\n'
        '      "description": "Clear description of what to do in this step.",\n'
        '      "dependencies": []\n'
        '    },\n'
        '    {\n'
        '      "number": 2,\n'
        '      "description": "Another step that may depend on step 1.",\n'
        '      "dependencies": [1]\n'
        '    },\n'
        '    ...\n'
        '  ]\n'
        '}\n'
        "\n"
        "Rules:\n"
        "- Each step should be achievable with the available tools\n"
        "- Include dependencies when a step needs results from a previous step\n"
        "- Keep the plan concise (3-7 steps is usually optimal)\n"
        "- Make sure the final step produces the answer the user wants\n"
        "\n"
        "Now create the plan for the goal above."
    )


# ====================================================================
# Part 3: The Planner
# ====================================================================

def _extract_json(text: str) -> dict:
    """
    Extract a JSON object from text that may have surrounding prose.
    
    This is simpler than parse_response - it just finds the first { and last }
    and tries to parse what's between them. The planner returns pure JSON,
    not an action/thought/answer structure.
    """
    if text is None:
        return {}
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return {}


def create_plan(goal: str, model=None) -> Plan:
    """
    Ask the LLM to create a plan for the given goal.
    
    The planner is a single-shot call - it does not use tools, it just
    thinks about the goal and produces a structured plan.
    """
    model = model or get_model()
    tools_instruction = generate_tools_instruction()
    prompt = build_planner_prompt(goal, tools_instruction)

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0.3,  # Lower for deterministic planning
    )
    raw = response.choices[0].message.content
    parsed = _extract_json(raw)

    summary = parsed.get("summary", "No summary provided")

    raw_steps = parsed.get("steps", [])
    steps = []
    for step_data in raw_steps:
        step = PlanStep(
            number=step_data.get("number", 0),
            description=step_data.get("description", ""),
            dependencies=step_data.get("dependencies", []),
        )
        steps.append(step)

    plan = Plan(summary=summary, steps=steps)

    print("")
    print("--- PLAN CREATED ---")
    print("Summary: " + plan.summary)
    for step in plan.steps:
        dep_str = " (depends on: " + str(step.dependencies) + ")" if step.dependencies else ""
        print("  Step " + str(step.number) + ": " + step.description + dep_str)

    return plan


# ====================================================================
# Part 4: The Plan-and-Execute Agent
# ====================================================================

class PlanAndExecuteAgent:
    """
    Combines planning and execution.

    The flow:
    1. Create a plan for the goal
    2. For each step in the plan (in order):
       a. Create a focused sub-goal for this step
       b. Run a ReAct agent to execute the sub-goal
       c. Record the result
    3. Combine all results into a final answer
    """

    def __init__(self, model=None, max_steps_per_subtask: int = 8):  # Increased to 8 for complex subtasks
        self.model = model or get_model()
        self.max_steps_per_subtask = max_steps_per_subtask

    def run(self, goal: str) -> dict:
        # Phase 1: Create the plan
        plan = create_plan(goal, model=self.model)

        # Phase 2: Execute each step
        results = {}
        completed = []

        remaining = list(plan.steps)

        while remaining:
            ready_steps = []
            for step in remaining:
                if all(dep in completed for dep in step.dependencies):
                    ready_steps.append(step)

            if not ready_steps:
                print("[WARNING]: Deadlock detected - some steps cannot be completed")
                break

            current_step = ready_steps[0]
            remaining.remove(current_step)

            print("")
            print("--- Executing Step " + str(current_step.number) + ": " + current_step.description + " ---")

            sub_goal = self._build_sub_goal(goal, plan, current_step, results)
            sub_result = self._execute_sub_goal(sub_goal)

            results[current_step.number] = sub_result
            completed.append(current_step.number)

            print("  [Result of Step " + str(current_step.number) + " (first 100 chars)]: " + str(sub_result)[:100] + "...")

        # Phase 3: Synthesize the final answer
        final_answer = self._synthesize(goal, plan, results)

        summary = {
            "goal": goal,
            "plan": plan,
            "results": results,
            "final_answer": final_answer,
            "completed_steps": completed,
            "total_steps": len(plan.steps),
        }

        print("")
        print("=" * 60)
        print("PHASE 3: FINAL SYNTHESIS")
        print("=" * 60)
        # Handle Unicode characters that can't be printed on Windows
        safe_answer = str(final_answer)[:200].encode('ascii', errors='replace').decode('ascii')
        print("Final Answer: " + safe_answer + "...")

        return summary

    def _build_sub_goal(self, original_goal: str, plan, current_step, results: dict) -> str:
        sub_goal = "Original Goal: " + original_goal + "\n"
        sub_goal += "Plan Summary: " + plan.summary + "\n"
        sub_goal += "Your task: Execute Step " + str(current_step.number) + ": " + current_step.description + "\n"

        if current_step.dependencies:
            sub_goal += "Previous results you can use:\n"
            for dep_num in current_step.dependencies:
                if dep_num in results:
                    sub_goal += "  - Step " + str(dep_num) + " result: " + str(results[dep_num])[:200] + "\n"

        sub_goal += "Use the available tools to complete this step. Return a concise result."
        return sub_goal

    def _execute_sub_goal(self, sub_goal: str) -> str:
        agent = ReActAgent(max_steps=self.max_steps_per_subtask, model=self.model)
        result = agent.run(goal=sub_goal)
        return result.get("answer", "[No answer produced]")

    def _synthesize(self, goal: str, plan, results: dict) -> str:
        all_results = ""
        for step in plan.steps:
            res = str(results.get(step.number, "N/A"))[:300]
            all_results += "Step " + str(step.number) + ": " + res + "\n"

        synthesis_prompt = (
            "You are a synthesizer. Combine these results into a final answer.\n"
            "Original Goal: " + goal + "\n"
            "Plan Summary: " + plan.summary + "\n"
            "Results: " + all_results + "\n"
            "Create a clear final answer."
        )

        client = get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": synthesis_prompt}],
            max_tokens=2048,
            temperature=0.7,
        )
        return response.choices[0].message.content


# ====================================================================
# Part 5: Test
# ====================================================================

if __name__ == "__main__":
    agent = PlanAndExecuteAgent(max_steps_per_subtask=6)

    result = agent.run(
        goal="Research the following: 1) Find the speed of light from the knowledge base, "
             "2) Find the capital of France from the knowledge base, "
             "3) Calculate how many light-years it takes for light to travel from the sun to "
             "a point that far away using the python tool, 4) Write all findings to a report.txt"
    )
