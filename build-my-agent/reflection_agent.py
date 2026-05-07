"""
Notebook 07: Reflection and Self-Critique
===========================================

The core idea: after producing an answer, critique it against a rubric,
score it, and revise if it doesn't meet the quality threshold.

This catches errors that the plan and execution didn't foresee:
- Missing details that the user expected
- Inaccurate calculations that weren't caught
- Poor organization or unclear explanations
- Format mismatches with the requested output

Architecture:
1. Executor: The original agent (Plan-and-Execute or ReAct) produces a first-draft answer
2. Critic: A separate LLM call evaluates the answer on multiple dimensions
3. Scorer: Each dimension gets a 1-10 score, plus an overall score
4. Revisor: If the overall score is below threshold, a new answer is generated
5. Loop: Critic evaluates the revised answer, repeat until good enough or max iterations
"""

import json
from dataclasses import dataclass, field

from config import get_client, get_model


# ====================================================================
# Part 1: The Critique Data Structure
# ====================================================================

@dataclass
class Critique:
    """
    One critique on the answer.
    
    Each critique has:
    - category: What aspect is being criticized (accuracy, completeness, clarity, format)
    - issue: A clear description of the problem
    - suggestion: How to fix it
    """
    category: str
    issue: str
    suggestion: str


@dataclass
class CritiqueReport:
    """
    The full critique report from the Critic.
    
    Contains:
    - score: Overall 1-10 quality score
    - scores: Per-dimension scores (accuracy, completeness, clarity, format)
    - critiques: List of specific issues found
    - is_acceptable: Whether the overall score meets the threshold
    """
    score: float
    scores: dict
    critiques: list = field(default_factory=list)
    is_acceptable: bool = False

    def __str__(self) -> str:
        lines = [
            "=== CRITIQUE REPORT ==",
            "Overall Score: {}/10 ({})".format(self.score, "PASS" if self.is_acceptable else "FAIL"),
            "Dimension Scores:",
        ]
        for dim, dim_score in self.scores.items():
            lines.append("  - {}: {}".format(dim, dim_score))

        if self.critiques:
            lines.append("Issues Found:")
            for c in self.critiques:
                lines.append("  - [{}] {}".format(c.category, c.issue))
                lines.append("    Suggestion: " + c.suggestion)
        else:
            lines.append("No issues found.")

        return "\n".join(lines)


# ====================================================================
# Part 2: The Critic Prompt (NATURAL style - works with this LLM)
# ====================================================================

def build_critic_prompt(goal: str, answer: str, threshold: float) -> str:
    """
    Build a natural-language prompt for the critic.
    
    The model responds better to conversational instructions than rigid JSON schemas.
    We ask for a structured text response and parse it programmatically.
    """
    return (
        "You are a quality reviewer. Evaluate the following answer against the original goal.\n\n"
        "## Original Goal\n"
        + goal + "\n\n"
        "## Answer to Evaluate\n"
        + answer + "\n\n"
        "## Your Task\n"
        "Score the answer on 4 dimensions (1-10), then list any issues found.\n\n"
        "1. Accuracy: Are the facts correct? Are calculations sound?\n"
        "2. Completeness: Does it address ALL parts of the goal? Nothing missing?\n"
        "3. Clarity: Is it well-organized, well-written, and easy to understand?\n"
        "4. Format: Does it follow the requested format (if any)?\n\n"
        "For each dimension, give a score (1-10) and a brief reason.\n\n"
        "Then, list any issues. For each issue, state the category (accuracy/completeness/clarity/format),\n"
        "what is wrong, and how to fix it.\n\n"
        "Finally, calculate the overall score as the average of the 4 dimension scores.\n"
        "State clearly whether the overall score meets the threshold of {}/10.\n\n"
        "Return ONLY a JSON object with this structure (no markdown, no extra text):\n"
        '{{"score": 7.5, "scores": {{"accuracy": 8, "completeness": 6, "clarity": 9, "format": 7}}, "critiques": [{{"category": "completeness", "issue": "...", "suggestion": "..."}}]}}'
    ).format(threshold)


# ====================================================================
# Part 3: The Critic
# ====================================================================

def _extract_json(text) -> dict:
    """Simple JSON extractor that handles the model's actual output style."""
    if text is None:
        return {}
    text = str(text).strip()
    
    # Try to find the JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    
    candidate = text[start:end+1]
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Try common failure patterns - remove markdown fences
    for marker in ["```json", "```", "~~~"]:
        text2 = text.replace(marker, "")
        start2 = text2.find("{")
        end2 = text2.rfind("}")
        if start2 >= 0 and end2 > start2:
            try:
                return json.loads(text2[start2:end2+1])
            except (json.JSONDecodeError, ValueError):
                pass
    
    return {}


def critique_answer(goal: str, answer: str, threshold: float = 7.0, model=None) -> CritiqueReport:
    """
    Call the LLM to critique the answer.
    
    The critic evaluates on 4 dimensions and returns a structured report.
    The report includes the overall score and whether it meets the threshold.
    """
    model = model or get_model()
    prompt = build_critic_prompt(goal, answer, threshold)

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.3,  # Deterministic evaluation
    )

    raw = response.choices[0].message.content
    if raw is None:
        raw = ""
    
    parsed = _extract_json(raw)
    
    # Debug: if parsing still fails, fall back to intelligent defaults
    if not parsed or "score" not in parsed:
        # The model gave a text response, try to extract scores from it
        return _parse_natural_response(raw, goal, answer, threshold)

    # Extract the overall score
    overall_score = float(parsed.get("score", 0.0))
    
    # Extract dimension scores
    dim_scores = parsed.get("scores", {})
    # Convert to proper types
    clean_scores = {}
    for k, v in dim_scores.items():
        clean_scores[k] = float(v) if v else 0.0
    
    # Extract critiques
    raw_critiques = parsed.get("critiques", [])
    critiques = []
    for item in (raw_critiques or []):
        if isinstance(item, dict):
            critique = Critique(
                category=item.get("category", "unknown") or "unknown",
                issue=item.get("issue", "No details provided") or "No details provided",
                suggestion=item.get("suggestion", "N/A") or "N/A",
            )
            critiques.append(critique)

    is_acceptable = overall_score >= threshold

    return CritiqueReport(
        score=overall_score,
        scores=clean_scores,
        critiques=critiques,
        is_acceptable=is_acceptable,
    )


def _parse_natural_response(raw, goal, answer, threshold):
    """
    Fallback: if the model gives a natural language response instead of JSON,
    try to extract the key information and construct a reasonable report.
    """
    import re
    
    raw_str = raw or ""
    
    # Look for score patterns like "Score: X/10" or "Overall: X/10"
    score_match = re.search(r'(?:score|overall)[:\s]*(\d+\.?\d*)[/\s]*(?:out of)?\s*10', raw_str, re.I)
    if not score_match:
        score_match = re.search(r'(\d+\.?\d*)/10', raw_str)
    
    overall_score = float(score_match.group(1)) if score_match else 5.0
    
    # Look for dimension scores
    scores = {"accuracy": 5.0, "completeness": 5.0, "clarity": 5.0, "format": 5.0}
    for dim in scores:
        pattern = re.search(r'(?:' + dim + r')[:\s]*(\d+\.?\d*)[/\s]*(?:out of)?\s*10', raw_str, re.I)
        if pattern:
            scores[dim] = float(pattern.group(1))
    
    # Extract issues from the text
    critiques = []
    # Look for problem descriptions
    lines = raw_str.split("\n")
    for line in lines:
        line_lower = line.lower().strip()
        if any(word in line_lower for word in ["missing", "incomplete", "incorrect", "not follow", "lacks", "no "]):
            # This line describes an issue, extract it
            clean_line = line.strip().lstrip("-•*").strip()
            if clean_line and len(clean_line) > 10:
                critiques.append(Critique(
                    category="completeness" if "missing" in line_lower or "incomplete" in line_lower else "accuracy",
                    issue=clean_line[:200],
                    suggestion="Address this issue in the next revision"
                ))
    
    # If no critiques found but score is low, add a generic one
    if not critiques and overall_score < 7.0:
        critiques.append(Critique(
            category="completeness",
            issue="The answer may be incomplete or lack depth based on the low score",
            suggestion="Review the original goal and ensure all aspects are covered"
        ))
    
    is_acceptable = overall_score >= threshold
    
    return CritiqueReport(
        score=overall_score,
        scores=scores,
        critiques=critiques,
        is_acceptable=is_acceptable,
    )


# ====================================================================
# Part 4: The Revision Prompt
# ====================================================================

def build_revision_prompt(goal: str, answer: str, report: CritiqueReport, threshold: float) -> str:
    """
    Build the prompt that tells the LLM to revise the answer.

    The reviser's job:
    1. Read the original goal
    2. Read the current (flawed) answer
    3. Read the critique report with specific issues
    4. Produce an improved answer that addresses all the issues

    This is like a teacher giving feedback on an essay, and the student rewriting it.
    """
    # Build the issues text from the critique report
    issues_text = ""
    for c in report.critiques:
        issues_text += "- [{}] {}\n  Suggestion: {}\n\n".format(c.category, c.issue, c.suggestion)

    if not issues_text:
        issues_text = "No specific issues. The answer met the quality threshold."

    prompt = (
        "You are a reviser. Improve the following answer based on the critique feedback below.\n\n"
        "## Original Goal\n"
        + goal + "\n\n"
        "## Current Answer (needs improvement)\n"
        + answer + "\n\n"
        "## Critique Report\n"
        "Overall Score: {}/10\n".format(report.score)
        + "Threshold: {}/10 (needs to meet this)\n\n".format(threshold)
        + "Dimension Scores:\n"
    )
    
    for dim, dim_score in report.scores.items():
        prompt += "  - {}: {}/10\n".format(dim, dim_score)

    prompt += "\nIssues to Address:\n" + issues_text
    prompt += "\nProduce a revised answer that fixes ALL the issues listed above. "
    prompt += "Make it complete, accurate, clear, and well-formatted according to the original goal.\n\n"
    prompt += "Return ONLY the revised answer, no extra commentary."

    return prompt


# ====================================================================
# Part 5: The Revision
# ====================================================================

def revise_answer(goal: str, answer: str, report: CritiqueReport, threshold: float, model=None) -> str:
    """
    Call the LLM to revise the answer based on the critique.

    The reviser reads the original goal, the current answer, and the critique,
    then produces an improved version.
    """
    model = model or get_model()
    prompt = build_revision_prompt(goal, answer, report, threshold)

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
    )

    content = response.choices[0].message.content
    if content is None:
        return answer  # Fallback to original if revision fails
    
    # Clean up: remove any "Here's the revised answer:" preamble
    content = content.strip()
    if content.startswith("Here's the revised answer"):
        content = content[content.find("\n\n"):]
    
    return content


# ====================================================================
# Part 6: The Reflection Agent
# ====================================================================

class ReflectionAgent:
    """
    Wraps any agent with a reflection loop.

    The cycle:
    1. Run the inner agent (e.g., Plan-and-Execute) to get a first-draft answer
    2. Critique the answer
    3. If score >= threshold: accept
    4. If score < threshold: revise and go to step 2
    5. Repeat until acceptable or max_iterations reached

    This is the "inner loop" of quality improvement.
    """

    def __init__(
        self,
        inner_agent,
        threshold: float = 7.0,
        max_iterations: int = 3,
        model=None,
    ):
        self.inner_agent = inner_agent
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.model = model or get_model()

    def run(self, goal: str) -> dict:
        """
        Run the inner agent, then reflect and revise until the answer is good enough.
        """
        # Step 1: Run the inner agent to get the first draft
        print("=" * 60)
        print("PHASE 1: RUNNING INNER AGENT (First Draft)")
        print("=" * 60)

        inner_result = self.inner_agent.run(goal)
        answer = inner_result.get("final_answer", "No answer produced")

        # Step 2-4: Reflect and revise loop
        history = []
        current_answer = answer
        iteration = 0

        print("\n" + "=" * 60)
        print("PHASE 2: REFLECTION LOOP")
        print("=" * 60)

        while iteration < self.max_iterations:
            iteration += 1
            print("\n--- Iteration {} ---".format(iteration))

            # Critique the current answer
            report = critique_answer(goal, current_answer, self.threshold, self.model)
            history.append({
                "iteration": iteration,
                "score": report.score,
                "is_acceptable": report.is_acceptable,
                "issues": len(report.critiques),
            })

            safe_report = str(report).encode('ascii', errors='replace').decode('ascii')
            print(safe_report)

            # Check if acceptable
            if report.is_acceptable:
                print("\n  [ACCEPTED] Score {}/10 meets threshold of {}/10".format(report.score, self.threshold))
                break

            # Not acceptable - revise
            print("\n  [REVISING] Score {}/10 below threshold of {}/10".format(report.score, self.threshold))
            current_answer = revise_answer(goal, current_answer, report, self.threshold, self.model)
            safe_answer = str(current_answer)[:100].encode('ascii', errors='replace').decode('ascii')
            print("  [Revision complete] (first 100 chars): " + safe_answer + "...")

        # Final result
        final_result = {
            "goal": goal,
            "initial_answer": answer,
            "final_answer": current_answer,
            "reflection_history": history,
            "iterations": iteration,
            "max_iterations": self.max_iterations,
            "threshold": self.threshold,
            "inner_agent_result": inner_result,
        }

        print("\n" + "=" * 60)
        print("PHASE 3: FINAL RESULT")
        print("=" * 60)
        final_safe = str(current_answer)[:300].encode('ascii', errors='replace').decode('ascii')
        print("Final Answer: " + final_safe + "...")

        return final_result


# ====================================================================
# Part 7: Test
# ====================================================================

if __name__ == "__main__":
    from plan_agent import PlanAndExecuteAgent

    # Create the inner agent (Plan-and-Execute)
    inner_agent = PlanAndExecuteAgent(max_steps_per_subtask=6)

    # Wrap it with the reflection layer
    agent = ReflectionAgent(
        inner_agent=inner_agent,
        threshold=7.0,
        max_iterations=3,
    )

    # Run with a complex goal that would benefit from reflection
    result = agent.run(
        goal="Find the speed of light and the capital of France from the knowledge base. "
             "Calculate how many light-seconds it takes for light to travel from the Sun to Earth. "
             "Write a summary report to report.txt with all findings."
    )

    print("\nReflection completed in {} iteration(s)".format(result["iterations"]))
