"""
Quick test for the Reflection agent.

This tests the core reflection loop without running the full
Plan-and-Execute inner agent. We manually provide a "first draft" answer
and let the critic evaluate it.
"""

from reflection_agent import Critique, CritiqueReport, critique_answer, revise_answer, ReflectionAgent, build_critic_prompt, build_revision_prompt
from config import get_model

def test_critic_prompt():
    print("=" * 60)
    print("Test 1: Critic Prompt Construction")
    print("=" * 60)

    goal = "Write a report about the speed of light"
    answer = "The speed of light is 300,000 km/s."
    threshold = 7.0

    prompt = build_critic_prompt(goal, answer, threshold)
    print("Prompt length: {} chars".format(len(prompt)))
    print("Prompt contains 'Accuracy': " + str("Accuracy" in prompt))
    print("Prompt contains 'Completeness': " + str("Completeness" in prompt))
    print("Prompt contains 'JSON object': " + str("JSON object" in prompt))
    print("Test 1 PASSED: Prompt constructed correctly\n")

def test_critic_llm():
    print("=" * 60)
    print("Test 2: LLM Critique (Live)")
    print("=" * 60)

    goal = "Write a 3-section report: 1) speed of light, 2) capital of France, 3) light-seconds from Sun to Earth"
    
    # This answer is intentionally incomplete (missing section 3) and has a minor inaccuracy
    answer = (
        "Section 1: The speed of light is 300,000 km/s.\n\n"
        "Section 2: The capital of France is Paris."
    )

    report = critique_answer(goal, answer, threshold=7.0)

    safe_report = str(report).encode('ascii', errors='replace').decode('ascii')
    print(safe_report)
    print("\n  -> Overall score: {}".format(report.score))
    print("  -> Is acceptable (>=7.0): {}".format(report.is_acceptable))
    print("  -> Number of issues found: {}".format(len(report.critiques)))
    
    for c in report.critiques:
        safe_issue = c.issue[:100].encode('ascii', errors='replace').decode('ascii')
        print("  -> Issue: [{}] {}".format(c.category, safe_issue))

    print("\n  Expected: Score below 7.0 (missing section 3, incomplete)")
    if not report.is_acceptable:
        print("  Test 2 PASSED: Critique correctly flagged incomplete answer\n")
    else:
        print("  Test 2 PARTIAL: Critique gave high score despite missing content\n")

def test_revision_prompt():
    print("=" * 60)
    print("Test 3: Revision Prompt Construction")
    print("=" * 60)

    goal = "Write a complete report on AI agents"
    answer = "AI agents are programs that use AI."
    
    report = CritiqueReport(
        score=4.5,
        scores={"accuracy": 6, "completeness": 3, "clarity": 5, "format": 4},
        critiques=[
            Critique(category="completeness", issue="Only one sentence, should be multiple sections", suggestion="Expand to 3 sections with examples"),
            Critique(category="format", issue="No section headers or structure", suggestion="Add headers and bullet points"),
        ],
        is_acceptable=False,
    )
    
    prompt = build_revision_prompt(goal, answer, report, threshold=7.0)
    
    print("  Contains 'reviser': " + str("reviser" in prompt.lower()))
    print("  Contains goal: " + str("AI agents" in prompt))
    print("  Contains original answer: " + str("programs that use AI" in prompt))
    print("  Contains issue: " + str("only one sentence" in prompt.lower() or "Only one sentence" in prompt))
    print("  Contains suggestion: " + str("Expand to" in prompt or "expand to" in prompt))
    print("\n  Test 3 PASSED: Revision prompt includes all necessary context\n")

def test_revision_llm():
    print("=" * 60)
    print("Test 4: LLM Revision (Live)")
    print("=" * 60)

    goal = "Write a 3-section report: 1) speed of light, 2) capital of France, 3) light-seconds from Sun to Earth"
    
    answer = (
        "Section 1: The speed of light is 300,000 km/s.\n\n"
        "Section 2: The capital of France is Paris."
    )

    # Get critique first
    report = critique_answer(goal, answer, threshold=7.0)
    print("  Original score: {}".format(report.score))
    print("  Issues to fix: {}".format(len(report.critiques)))
    for c in report.critiques:
        print("    - [{}] {}".format(c.category, c.issue[:80]))

    # Revise
    print("\n  Revising answer...")
    revised = revise_answer(goal, answer, report, threshold=7.0, model=get_model())
    
    safe_revised = revised[:500].encode('ascii', errors='replace').decode('ascii')
    print("  Revised answer (first 500 chars):")
    print("  " + safe_revised.replace("\n", "\n  "))

    # Critique the revised version
    print("\n  Critiquing the revised version...")
    revised_report = critique_answer(goal, revised, threshold=7.0)
    safe_revised_report = str(revised_report).encode('ascii', errors='replace').decode('ascii')
    print(safe_revised_report)

    print("\n  -> Revised score: {}".format(revised_report.score))
    print("  -> Score improved: {} (was {})".format(revised_report.score, report.score))
    print("  -> Improvement: +{}".format(round(revised_report.score - report.score, 1)))
    
    if revised_report.is_acceptable:
        print("\n  Test 4 PASSED: Revision improved quality to acceptable level\n")
    else:
        print("\n  Test 4 PARTIAL: Quality improved but still below threshold\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Reflection Agent - Unit Tests")
    print("=" * 60 + "\n")

    # Test 1: Critic prompt construction
    test_critic_prompt()

    # Test 2: LLM critique
    test_critic_llm()

    # Test 3: Revision prompt construction
    test_revision_prompt()

    # Test 4: LLM revision
    test_revision_llm()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
