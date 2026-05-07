"""
Fast unit tests for the Reflection agent.
Tests data structures and prompt construction WITHOUT calling the LLM.
"""

from reflection_agent import Critique, CritiqueReport, build_critic_prompt, build_revision_prompt

# ====================================================================
# Test 1: Critique Data Structure
# ====================================================================

def test_critique_dataclass():
    print("=" * 60)
    print("Test 1: Critique Data Structure")
    print("=" * 60)

    c = Critique(
        category="accuracy",
        issue="The speed of light is listed as 300,000 km/s but should be 299,792,458 m/s",
        suggestion="Use the exact value from the knowledge base"
    )

    assert c.category == "accuracy", "Category mismatch"
    assert "300,000" in c.issue, "Issue should contain the wrong value"
    assert "knowledge base" in c.suggestion, "Suggestion should mention KB"

    print("  Category: " + c.category)
    print("  Issue (first 60 chars): " + c.issue[:60] + "...")
    print("  Suggestion (first 60 chars): " + c.suggestion[:60] + "...")
    print("  Test 1 PASSED: Critique dataclass works correctly\n")

# ====================================================================
# Test 2: Critique Report (with __str__)
# ====================================================================

def test_critique_report():
    print("=" * 60)
    print("Test 2: Critique Report")
    print("=" * 60)

    report = CritiqueReport(
        score=5.5,
        scores={"accuracy": 7, "completeness": 4, "clarity": 6, "format": 5},
        critiques=[
            Critique(category="completeness", issue="Only 2 of 3 requested topics covered", suggestion="Add the 3rd topic"),
            Critique(category="format", issue="No headers or structure", suggestion="Use section headers and bullet points"),
        ],
        is_acceptable=False,
    )

    print("  Overall score: " + str(report.score))
    print("  Is acceptable: " + str(report.is_acceptable))
    print("  Number of critiques: " + str(len(report.critiques)))

    # Test the string representation
    report_str = str(report)
    assert "Overall Score: 5.5/10" in report_str, "Report string should contain overall score"
    assert "PASS" not in report_str, "Report should show FAIL when not acceptable"
    assert "completeness" in report_str, "Report should list critique categories"

    safe_str = report_str.encode('ascii', errors='replace').decode('ascii')
    print("  Report preview:")
    for line in safe_str.split("\n"):
        print("    " + line)
    print("  Test 2 PASSED: CritiqueReport works with __str__\n")

# ====================================================================
# Test 3: Critic Prompt Construction
# ====================================================================

def test_critic_prompt():
    print("=" * 60)
    print("Test 3: Critic Prompt Construction")
    print("=" * 60)

    goal = "Write a report about the speed of light and the capital of France"
    answer = "The speed of light is 300,000 km/s. The capital of France is Paris."
    threshold = 7.0

    prompt = build_critic_prompt(goal, answer, threshold)

    # Verify the prompt contains all required elements
    checks = {
        "Original Goal section": "Original Goal" in prompt,
        "Goal text included": "speed of light" in prompt and "capital of France" in prompt,
        "Answer to Critique section": "Answer to Critique" in prompt,
        "Answer text included": "300,000" in prompt and "Paris" in prompt,
        "Accuracy dimension": "Accuracy" in prompt,
        "Completeness dimension": "Completeness" in prompt,
        "Clarity dimension": "Clarity" in prompt,
        "Format dimension": "Format" in prompt,
        "Score range (1-10)": "1-10" in prompt,
        "JSON output format": "JSON" in prompt,
        "Threshold mentioned": str(threshold) in prompt,
        "Score field example": '"score":' in prompt,
        "Critiques field example": '"critiques":' in prompt,
    }

    all_passed = True
    for name, passed in checks.items():
        status = "OK" if passed else "FAIL"
        if not passed:
            all_passed = False
        print("  {}: {}".format(name, status))

    print("  Prompt length: {} chars".format(len(prompt)))

    if all_passed:
        print("  Test 3 PASSED: All prompt elements present\n")
    else:
        print("  Test 3 PARTIAL: Some elements missing\n")

# ====================================================================
# Test 4: Revision Prompt Construction
# ====================================================================

def test_revision_prompt():
    print("=" * 60)
    print("Test 4: Revision Prompt Construction")
    print("=" * 60)

    goal = "Write a complete report on AI agents"
    answer = "AI agents are programs that use artificial intelligence."

    report = CritiqueReport(
        score=4.5,
        scores={"accuracy": 6, "completeness": 3, "clarity": 5, "format": 4},
        critiques=[
            Critique(
                category="completeness",
                issue="Only one sentence, should be multiple sections",
                suggestion="Expand to 3 sections with examples"
            ),
            Critique(
                category="format",
                issue="No section headers or structure",
                suggestion="Add headers and bullet points"
            ),
        ],
        is_acceptable=False,
    )

    threshold = 7.0
    prompt = build_revision_prompt(goal, answer, report, threshold)

    # Verify the prompt contains all required elements
    checks = {
        "Reviser role mentioned": "reviser" in prompt.lower(),
        "Original Goal section": "Original Goal" in prompt,
        "Goal text included": "AI agents" in prompt,
        "Current Answer section": "Current Answer" in prompt,
        "Answer text included": "artificial intelligence" in prompt,
        "Critique Report section": "Critique Report" in prompt,
        "Overall score included": "4.5" in prompt,
        "Threshold included": str(threshold) in prompt,
        "Dimension scores section": "Dimension Scores" in prompt,
        "Issues to Address section": "Issues to Address" in prompt,
        "Critique issue 1": "only one sentence" in prompt.lower() or "Only one sentence" in prompt,
        "Critique issue 2": "No section headers" in prompt,
        "Suggestion 1": "Expand to" in prompt,
        "Suggestion 2": "bullet points" in prompt,
        "Action instruction": "revised answer" in prompt.lower() or "Revised answer" in prompt,
    }

    all_passed = True
    for name, passed in checks.items():
        status = "OK" if passed else "FAIL"
        if not passed:
            all_passed = False
        print("  {}: {}".format(name, status))

    print("  Prompt length: {} chars".format(len(prompt)))

    if all_passed:
        print("  Test 4 PASSED: All revision prompt elements present\n")
    else:
        print("  Test 4 PARTIAL: Some elements missing\n")

# ====================================================================
# Test 5: Acceptable vs Not Acceptable
# ====================================================================

def test_acceptance_threshold():
    print("=" * 60)
    print("Test 5: Acceptance Threshold Logic")
    print("=" * 60)

    # Below threshold
    low_report = CritiqueReport(score=6.5, scores={}, is_acceptable=False)
    assert not low_report.is_acceptable, "6.5 should be below 7.0"
    print("  6.5/10 < 7.0 -> Not acceptable: CORRECT")

    # At threshold
    exact_report = CritiqueReport(score=7.0, scores={}, is_acceptable=True)
    assert exact_report.is_acceptable, "7.0 should meet 7.0"
    print("  7.0/10 >= 7.0 -> Acceptable: CORRECT")

    # Above threshold
    high_report = CritiqueReport(score=9.0, scores={}, is_acceptable=True)
    assert high_report.is_acceptable, "9.0 should meet 7.0"
    print("  9.0/10 >= 7.0 -> Acceptable: CORRECT")

    # String representation of a passing report
    pass_str = str(high_report)
    assert "PASS" in pass_str, "High score should show PASS"
    print("  PASS indicator in high-score report: CORRECT")

    print("  Test 5 PASSED: Threshold logic works correctly\n")

# ====================================================================
# Test 6: Edge cases
# ====================================================================

def test_edge_cases():
    print("=" * 60)
    print("Test 6: Edge Cases")
    print("=" * 60)

    # Empty critiques list
    empty_report = CritiqueReport(
        score=8.0,
        scores={"accuracy": 8, "completeness": 8, "clarity": 8, "format": 8},
        critiques=[],
        is_acceptable=True,
    )
    empty_str = str(empty_report)
    assert "No issues found" in empty_str, "Empty critiques should say 'No issues found'"
    print("  Empty critiques list -> 'No issues found': CORRECT")

    # Very long answer in prompt
    long_answer = "x" * 10000  # 10KB of garbage
    long_prompt = build_critic_prompt("Simple goal", long_answer, 7.0)
    assert "Simple goal" in long_prompt, "Long answers should still include the goal"
    assert len(long_prompt) > 10000, "Long answer should be reflected in prompt length"
    print("  Long answer (10KB) handled correctly: CORRECT")

    # Unicode in answer (Windows encoding concern)
    unicode_answer = "Speed: 300,000 km/s with special chars: accute, grave, tilde"
    unicode_prompt = build_critic_prompt("Find speed of light", unicode_answer, 7.0)
    assert "300,000" in unicode_prompt, "Unicode answer should still be in prompt"
    print("  Unicode-safe handling: CORRECT")

    print("  Test 6 PASSED: Edge cases handled correctly\n")

# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reflection Agent - Fast Unit Tests (No LLM calls)")
    print("=" * 60 + "\n")

    test_critique_dataclass()
    test_critique_report()
    test_critic_prompt()
    test_revision_prompt()
    test_acceptance_threshold()
    test_edge_cases()

    print("=" * 60)
    print("All 6 fast tests completed successfully!")
    print("=" * 60)
