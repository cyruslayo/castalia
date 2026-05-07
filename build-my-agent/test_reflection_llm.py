"""
LLM validation tests for the Reflection agent.
Runs actual LLM calls to validate the critic, reviser, and reflection loop.
"""

from reflection_agent import critique_answer, revise_answer

# ====================================================================
# Test 1: Critic with intentional flaws
# ====================================================================

def test_critic_with_flaws():
    print("=" * 60)
    print("Test 1: LLM Critic (Intentional Flaws)")
    print("=" * 60)

    goal = (
        "Write a 3-section report: "
        "1) Speed of light, 2) Capital of France, "
        "3) Light-seconds from Sun to Earth. "
        "Present as bullet points with section headers."
    )

    # Intentionally incomplete (missing section 3) and wrong format (paragraphs, not bullets)
    answer = (
        "The speed of light is 300,000 km/s. "
        "The capital of France is Paris."
    )

    print("  Goal: 3-section report with bullet points")
    print("  Answer: 2 facts, no sections, no bullets, missing section 3")
    print("  Calling critic...")

    report = critique_answer(goal, answer, threshold=7.0)

    safe_report = str(report).encode('ascii', errors='replace').decode('ascii')
    print("\n  Critique Report:")
    for line in safe_report.split("\n"):
        print("    " + line)

    print("\n  -> Overall score: {}".format(report.score))
    print("  -> Is acceptable (>=7.0): {}".format(report.is_acceptable))
    print("  -> Issues found: {}".format(len(report.critiques)))

    for i, c in enumerate(report.critiques, 1):
        safe_issue = c.issue[:80].encode('ascii', errors='replace').decode('ascii')
        print("    Issue {}: [{}] {}".format(i, c.category, safe_issue))

    # Validate expectations
    assert report.score < 7.0, "Flawed answer should score below 7.0, got {}".format(report.score)
    assert not report.is_acceptable, "Should not be acceptable"
    assert len(report.critiques) > 0, "Should find issues"

    # Check that completeness was flagged (missing section 3 is a completeness issue)
    categories = [c.category.lower() for c in report.critiques]
    has_completeness_issue = "completeness" in categories
    has_format_issue = "format" in categories

    print("\n  Flagged completeness issue: {}".format(has_completeness_issue))
    print("  Flagged format issue: {}".format(has_format_issue))

    if has_completeness_issue or has_format_issue:
        print("\n  Test 1 PASSED: Critic correctly identified major flaws\n")
    else:
        print("\n  Test 1 PARTIAL: Critic flagged something, but not the expected dimensions\n")

    return report

# ====================================================================
# Test 2: Reviser fixes the flaws
# ====================================================================

def test_reviser_fixes_flaws(report):
    print("=" * 60)
    print("Test 2: LLM Reviser (Fix the Flaws)")
    print("=" * 60)

    goal = (
        "Write a 3-section report: "
        "1) Speed of light, 2) Capital of France, "
        "3) Light-seconds from Sun to Earth. "
        "Present as bullet points with section headers."
    )

    answer = (
        "The speed of light is 300,000 km/s. "
        "The capital of France is Paris."
    )

    print("  Original answer: 2 facts, no sections, no bullets, missing section 3")
    print("  Original score: {}".format(report.score))
    print("  Calling reviser with critique feedback...")

    revised = revise_answer(goal, answer, report, threshold=7.0)

    safe_revised = revised[:600].encode('ascii', errors='replace').decode('ascii')
    print("\n  Revised answer (first 600 chars):")
    for line in safe_revised.split("\n"):
        print("    " + line)

    # Check for improvements
    has_sections = "section" in revised.lower() or "section 1" in revised.lower() or "section 2" in revised.lower() or "section 3" in revised.lower()
    has_bullets = "-" in revised or "*" in revised or "\u2022" in revised  # bullet characters
    has_section3 = "earth" in revised.lower() or "sun" in revised.lower() or "light" in revised.lower()
    has_speed = "300,000" in revised or "299" in revised or "speed" in revised.lower()
    has_france = "paris" in revised.lower() or "france" in revised.lower() or "capital" in revised.lower()

    print("\n  Improvement checks:")
    print("  Has section structure: {}".format(has_sections))
    print("  Has bullet points: {}".format(has_bullets))
    print("  Has section 3 content (Earth/Sun/light-seconds): {}".format(has_section3))
    print("  Has speed of light: {}".format(has_speed))
    print("  Has France/capital: {}".format(has_france))
    print("  Revised length: {} chars (original: {})".format(len(revised), len(answer)))

    # The revised answer should be longer (more complete) and better structured
    is_longer = len(revised) > len(answer)
    has_more_content = is_longer and (has_sections or has_bullets)

    if has_more_content and (has_france and has_speed):
        print("\n  Test 2 PASSED: Reviser produced a longer, more structured answer\n")
    else:
        print("\n  Test 2 PARTIAL: Some improvements detected, but not all expected\n")

    return revised

# ====================================================================
# Test 3: Critic evaluates the revised answer
# ====================================================================

def test_critic_evaluates_revision(revised):
    print("=" * 60)
    print("Test 3: Critic Evaluates the Revision")
    print("=" * 60)

    goal = (
        "Write a 3-section report: "
        "1) Speed of light, 2) Capital of France, "
        "3) Light-seconds from Sun to Earth. "
        "Present as bullet points with section headers."
    )

    print("  Sending revised answer to critic for re-evaluation...")

    revised_report = critique_answer(goal, revised, threshold=7.0)

    safe_report = str(revised_report).encode('ascii', errors='replace').decode('ascii')
    print("\n  Revised Critique Report:")
    for line in safe_report.split("\n"):
        print("    " + line)

    return revised_report

# ====================================================================
# Test 4: Full reflection loop (2 iterations)
# ====================================================================

def test_full_reflection_loop():
    print("=" * 60)
    print("Test 4: Full Reflection Loop (2 iterations)")
    print("=" * 60)

    goal = (
        "Write a 3-section report: "
        "1) Speed of light, 2) Capital of France, "
        "3) Light-seconds from Sun to Earth. "
        "Present as bullet points with section headers."
    )

    answer = (
        "The speed of light is 300,000 km/s. "
        "The capital of France is Paris."
    )

    print("  Goal: 3-section report with bullet points")
    print("  First draft: 2 facts, no structure, missing section 3")
    print("  Threshold: 7.0/10, Max iterations: 2")

    history = []
    current_answer = answer
    threshold = 7.0
    max_iterations = 2

    for iteration in range(1, max_iterations + 1):
        print("\n  --- Iteration {} ---".format(iteration))

        report = critique_answer(goal, current_answer, threshold)
        history.append({
            "iteration": iteration,
            "score": report.score,
            "is_acceptable": report.is_acceptable,
            "issues": len(report.critiques),
        })

        safe_report = str(report).encode('ascii', errors='replace').decode('ascii')
        print("  Critique report:")
        for line in safe_report.split("\n"):
            print("    " + line)

        if report.is_acceptable:
            print("\n  [ACCEPTED on iteration {}]".format(iteration))
            break

        if iteration < max_iterations:
            print("\n  [REVISING - calling reviser...]")
            current_answer = revise_answer(goal, current_answer, report, threshold)
            safe_partial = str(current_answer)[:200].encode('ascii', errors='replace').decode('ascii')
            print("  [Revision done] (first 200 chars): " + safe_partial)

    print("\n  " + "=" * 50)
    print("  REFLECTION JOURNEY")
    print("=" * 50)

    for entry in history:
        status = "PASS" if entry["is_acceptable"] else "FAIL"
        print("  Iteration {}: {}/10 ({}), {} issues".format(
            entry["iteration"],
            entry["score"],
            status,
            entry["issues"]
        ))

    # Show improvement
    if len(history) >= 2:
        score_improvement = history[-1]["score"] - history[0]["score"]
        print("\n  Score improvement: {:.1f} -> {:.1f} (+{:+.1f})".format(
            history[0]["score"], history[-1]["score"], score_improvement
        ))

    has_improved = any(
        h["score"] > history[0]["score"] for h in history[1:]
    ) if len(history) > 1 else False

    print("\n  Test 4 Result:")
    print("  Total iterations: {}".format(len(history)))
    print("  Score improved from first to last: {}".format(has_improved))

    if has_improved:
        print("  Test 4 PASSED: Score improved through reflection loop\n")
    else:
        print("  Test 4 PARTIAL: Loop ran, but score trajectory needs review\n")

    return history

# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reflection Agent - LLM Validation Tests")
    print("(Real LLM calls to verify the implementation)")
    print("=" * 60 + "\n")

    print("  NOTE: These tests make actual LLM calls. Expect 30-120s total.\n")

    # Test 1: Critic identifies flaws
    report = test_critic_with_flaws()

    # Test 2: Reviser fixes the flaws
    revised = test_reviser_fixes_flaws(report)

    # Test 3: Critic evaluates the revision
    revised_report = test_critic_evaluates_revision(revised)

    # Show comparison
    print("\n  " + "=" * 50)
    print("  SCORE COMPARISON")
    print("=" * 50)
    print("  Original:  {}/10".format(report.score))
    print("  Revised:   {}/10 ({} {})".format(
        revised_report.score,
        "PASS" if revised_report.is_acceptable else "FAIL",
        "(above threshold)" if revised_report.is_acceptable else "(below threshold)"
    ))
    print("  Change:    +{:+.1f}".format(revised_report.score - report.score))

    # Test 4: Full reflection loop
    test_full_reflection_loop()

    print("\n" + "=" * 60)
    print("All LLM validation tests completed!")
    print("=" * 60)
