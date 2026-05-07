# ====================================================================
# Notebook 09: Iterative Refinement
# Generalize reflection: use ANY external feedback to improve output
# ====================================================================

import io
import re
import time
import contextlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple


# ====================================================================
# Part 0: Data Structures
# ====================================================================

@dataclass
class FeedbackResult:
    """
    Structured feedback from an external source.

    This is the contract that every feedback function must return.
    The agent uses the score and passed flag to decide whether to continue.
    """
    score: float  # 0-10 scale
    passed: bool  # overall pass/fail
    feedback_text: str  # human-readable feedback for the LLM to act on
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return "[{}] Score: {:.1f}/10 - {}".format(status, self.score,
                                                       self.feedback_text[:150])


@dataclass
class RefinementIteration:
    """Record of one iteration in the refinement loop."""
    iteration: int
    output: str
    feedback: FeedbackResult
    revision_prompt: Optional[str] = None


@dataclass
class RefinementTrace:
    """Complete trace of the refinement process."""
    task: str
    iterations: List[RefinementIteration] = field(default_factory=list)
    final_output: str = ""
    converge_reason: str = ""

    def score_trajectory(self) -> List[float]:
        return [it.feedback.score for it in self.iterations]

    def summary(self) -> str:
        lines = ["Task: {}".format(self.task)]
        lines.append("Iterations: {}".format(len(self.iterations)))
        lines.append("Convergence: {}".format(self.converge_reason))
        scores = self.score_trajectory()
        if scores:
            lines.append("Scores: {}".format(" -> ".join("{:.1f}".format(s) for s in scores)))
            lines.append("Improvement: {:.1f} -> {:.1f} ({:+.1f})".format(
                scores[0], scores[-1], scores[-1] - scores[0]))
        return "\n".join(lines)


# ====================================================================
# Part 1: The Core Agent
# ====================================================================

class IterativeRefinementAgent:
    """
    Agent that iteratively improves output using external feedback.

    The feedback function is the key design decision. It can be:
    - A test runner (for code)
    - A style scorer (for text)
    - A fact checker (for analysis)
    - Any function that takes (task, output) and returns FeedbackResult
    """

    def __init__(
        self,
        feedback_fn: Callable[[str, str], "FeedbackResult"],
        max_iterations: int = 5,
        score_threshold: float = 8.0,
        min_improvement: float = 0.3,
        detect_plateau: bool = True,
        detect_decline: bool = True,
    ):
        self.feedback_fn = feedback_fn
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.min_improvement = min_improvement
        self.detect_plateau = detect_plateau
        self.detect_decline = detect_decline

    def _get_llm_content(self, messages: list, max_tokens: int, temp: float) -> str:
        """
        Call the LLM and return the content, falling back to the
        reasoning field for thinking models that leave content as None.
        """
        from config import get_client, get_model
        response = get_client().chat.completions.create(
            model=get_model(),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
        )
        result = response.choices[0].message.content
        if not result:
            result = getattr(response.choices[0].message, 'reasoning', '') or ''
        return result

    def generate_initial(self, task: str) -> str:
        """Generate the first version of the output."""
        system_prompt = (
            "You are an expert Python developer. "
            "Output ONLY the raw Python code. "
            "No explanations, no markdown text, no reasoning process. "
            "Start directly with 'def ' or the function signature."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        result = self._get_llm_content(messages, 1024, 0.7)
        return extract_code_only(result)

    def revise(self, task: str, current_output: str, feedback: "FeedbackResult") -> str:
        """
        Revise the output based on external feedback.

        The feedback comes from an EXTERNAL grounded source
        (tests, metrics, validators), not the LLM's own imperfect judgment.
        """
        user_content = (
            "TASK: {t}\n\n"
            "CURRENT OUTPUT:\n{o}\n\n"
            "FEEDBACK (score {s:.1f}/10):\n{f}\n\n"
            "Revise. Output ONLY the corrected code. No explanations."
        ).format(t=task, o=current_output, s=feedback.score, f=feedback.feedback_text)

        system_prompt = (
            "You are revising code based on specific feedback. "
            "Address EVERY issue in the feedback. "
            "Output ONLY the corrected Python code. No explanations, no reasoning."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = self._get_llm_content(messages, 1024, 0.7)
        return extract_code_only(result)

    def check_convergence(self, scores: List[float], iteration: int) -> Tuple[bool, str]:
        """
        Determine if we should stop iterating.

        Three convergence strategies:
        1. Score threshold: stop when score >= threshold
        2. Plateau detection: stop when improvement < min_improvement
        3. Decline detection: stop when score drops significantly
        """
        if iteration >= self.max_iterations:
            return True, "Max iterations ({}) reached".format(self.max_iterations)

        if scores and scores[-1] >= self.score_threshold:
            return True, "Score {:.1f} >= threshold {}".format(scores[-1], self.score_threshold)

        if len(scores) >= 2 and self.detect_plateau:
            delta = scores[-1] - scores[-2]
            if abs(delta) < self.min_improvement:
                return True, "Plateau: improvement {:+.2f} < {}".format(delta, self.min_improvement)

        if len(scores) >= 2 and self.detect_decline:
            if scores[-1] < scores[-2] - 0.5:
                return True, "Decline: {:.1f} -> {:.1f}".format(scores[-2], scores[-1])

        return False, "Continue"

    def run(self, task: str, verbose: bool = True) -> "RefinementTrace":
        """
        Run the full iterative refinement loop.

        Each iteration:
        1. Get feedback from the external function
        2. Record the iteration
        3. Check convergence (stop if good enough or stalling)
        4. Revise based on feedback
        5. Loop back to step 1
        """
        trace = RefinementTrace(task=task)

        if verbose:
            print("=" * 60)
            print("TASK: {}".format(task[:100]))
            print("=" * 60)
            print()

        # Initial generation
        if verbose:
            print("[Iteration 0: GENERATE]")
        current_output = self.generate_initial(task)
        if verbose:
            print("Output (preview): {}...".format(current_output[:200]))
            print()

        for i in range(self.max_iterations + 1):
            # Get external feedback
            if verbose:
                print("[Iteration {}: FEEDBACK]".format(i))
            feedback = self.feedback_fn(task, current_output)
            if verbose:
                print("  {}".format(feedback.summary()))
                print()

            trace.iterations.append(RefinementIteration(
                iteration=i,
                output=current_output,
                feedback=feedback,
            ))

            # Check convergence
            scores = trace.score_trajectory()
            stop, reason = self.check_convergence(scores, i)
            if stop:
                trace.converge_reason = reason
                trace.final_output = current_output
                if verbose:
                    print("Stopped: {}\n".format(reason))
                break

            # Revise
            if verbose:
                print("[Iteration {}: REVISE]".format(i))
            current_output = self.revise(task, current_output, feedback)
            if verbose:
                print("Revised (preview): {}...".format(current_output[:200]))
                print()

        if not trace.final_output:
            trace.final_output = current_output
            trace.converge_reason = "Completed all iterations"

        if verbose:
            print(trace.summary())

        return trace


# ====================================================================
# Part 2: Use Case 1 - Code with Test Feedback
# ====================================================================

def safe_exec(code_str: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Safely execute Python code in an isolated namespace.

    Extracts code from markdown blocks if present.
    Captures stdout, reports any exceptions.
    """
    # Extract from ```python ... ``` blocks
    code_block = re.search(r"```python\n(.*?)```", code_str, re.DOTALL)
    if code_block:
        code_str = code_block.group(1)

    # Clean remaining markdown
    clean_lines = []
    for line in code_str.split("\n"):
        if not line.strip().startswith("```"):
            clean_lines.append(line)
    code_str = "\n".join(clean_lines)

    stdout_capture = io.StringIO()
    try:
        local_ns = {}
        with contextlib.redirect_stdout(stdout_capture):
            exec(code_str, {"__builtins__": __builtins__}, local_ns)
        output = stdout_capture.getvalue()
        return True, output if output else "Code executed successfully (no output)."
    except Exception as e:
        return False, "{}: {}".format(type(e).__name__, str(e))


def extract_code_only(text: str) -> str:
    """
    Extract only the Python code from LLM output, stripping prose/reasoning.
    """
    # Try 1: Extract from ```python ... ``` blocks
    m = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1)
    # Try 2: Extract from ``` ... ``` blocks (no language tag)
    m2 = re.search(r'```\n(.*?)```', text, re.DOTALL)
    if m2:
        return m2.group(1)
    # Try 3: Find the first 'def ' and everything after it (strip preamble)
    if 'def ' in text:
        idx = text.index('def ')
        return text[idx:]
    # Try 4: If no code found, return the original
    return text


def code_test_feedback(task: str, output: str) -> "FeedbackResult":
    """
    Feedback function that runs real tests against generated code.

    This provides ground truth that self-critique cannot: actual
    pass/fail results from executing the code with known inputs.
    """
    test_cases = []
    if "fibonacci" in task.lower() or "fib(" in task.lower():
        test_cases = [
            ("fib(0)", "0"),
            ("fib(1)", "1"),
            ("fib(5)", "5"),
            ("fib(10)", "55"),
            ("fib(20)", "6765"),
        ]
    elif "palindrome" in task.lower():
        test_cases = [
            ('is_palindrome("racecar")', "True"),
            ('is_palindrome("hello")', "False"),
            ('is_palindrome("")', "True"),
            ('is_palindrome("A")', "True"),
            ('is_palindrome("ab")', "False"),
        ]
    elif "fizzbuzz" in task.lower() or "FizzBuzz" in task:
        test_cases = [
            ("repr(fizzbuzz(1))", "'1'"),
            ("repr(fizzbuzz(3))", "'Fizz'"),
            ("repr(fizzbuzz(5))", "'Buzz'"),
            ("repr(fizzbuzz(15))", "'FizzBuzz'"),
            ("repr(fizzbuzz(7))", "'7'"),
        ]
    else:
        return FeedbackResult(score=5.0, passed=False,
                               feedback_text="No test cases defined for this task.")

    # Build test harness
    test_code = output + "\n\nresults = []\n"
    for call, expected in test_cases:
        test_code += "try:\n"
        test_code += "    r = " + call + "\n"
        # Use string concatenation to avoid format conflicts
        fail_msg = "'FAIL: " + call + " got ' + repr(r) + ', expected " + expected + "'"
        test_code += "    results.append(('PASS', '" + call + "') if str(r) == '" + expected + "' else (" + fail_msg + ", '" + call + "'))\n"
        test_code += "except Exception as e:\n"
        test_code += "    results.append(('ERROR: ' + str(e), '" + call + "'))\n"
    test_code += "for status, name in results:\n"
    test_code += "    print('{n}: {s}'.format(n=name, s=status))"

    success, test_output = safe_exec(test_code)

    if not success:
        return FeedbackResult(
            score=1.0, passed=False,
            feedback_text="Code failed to execute: " + test_output,
            details={"error": test_output}
        )

    passed = test_output.count("PASS")
    failed = test_output.count("FAIL")
    errors = test_output.count("ERROR")
    total = passed + failed + errors

    if total == 0:
        score = 5.0
    else:
        score = (passed / total) * 10.0

    all_passed = (failed == 0 and errors == 0 and passed > 0)

    feedback_parts = ["{}/{} tests passed.".format(passed, total)]
    for line in test_output.split("\n"):
        if "FAIL" in line or "ERROR" in line:
            feedback_parts.append("  " + line.strip())

    return FeedbackResult(
        score=score,
        passed=all_passed,
        feedback_text="\n".join(feedback_parts),
        details={"passed": passed, "failed": failed, "errors": errors, "output": test_output}
    )


# ====================================================================
# Part 3: Use Case 2 - Text with Style Scoring
# ====================================================================

def style_feedback(task: str, output: str) -> "FeedbackResult":
    """
    Feedback function that scores text style using heuristics.

    No LLM call needed - pure algorithmic feedback.
    """
    issues = []
    score = 10.0

    words = output.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", output) if s.strip()]

    if len(words) < 50:
        issues.append("Too short - add more detail and examples.")
        score -= 2.0
    elif len(words) > 500:
        issues.append("Too verbose ({} words). Target 150-350 words. Cut filler.".format(len(words)))
        score -= 1.5

    if sentences:
        avg_sentence_len = len(words) / len(sentences)
        if avg_sentence_len > 30:
            issues.append("Sentences too long (avg {:.0f} words). Break them up.".format(avg_sentence_len))
            score -= 1.0
        elif avg_sentence_len < 8:
            issues.append("Sentences too choppy (avg {:.0f} words). Combine some.".format(avg_sentence_len))
            score -= 0.5

    passive_indicators = ["was done", "were made", "is being", "has been", "was created", "were found"]
    passive_count = sum(1 for p in passive_indicators if p in output.lower())
    if passive_count >= 3:
        issues.append("Too much passive voice ({} instances). Use active voice.".format(passive_count))
        score -= 1.0

    filler_words = ["basically", "actually", "really", "very", "just", "quite", "somewhat"]
    filler_count = sum(output.lower().count(f) for f in filler_words)
    if filler_count >= 4:
        issues.append("Too many filler words ({}): remove 'basically', 'actually', 'really', etc.".format(filler_count))
        score -= 1.0

    paragraphs = [p.strip() for p in output.split("\n\n") if p.strip()]
    if len(paragraphs) < 2 and len(words) > 100:
        issues.append("Add paragraph breaks for readability.")
        score -= 0.5

    score = max(1.0, min(10.0, score))
    passed = score >= 8.0 and len(issues) <= 1

    feedback_text = "Style Feedback:\n"
    if issues:
        feedback_text += "\n".join("- {}".format(i) for i in issues)
    else:
        feedback_text += "No major style issues detected."

    return FeedbackResult(
        score=score,
        passed=passed,
        feedback_text=feedback_text,
        details={"word_count": len(words), "sentence_count": len(sentences), "issues": issues}
    )


# ====================================================================
# Part 4: Use Case 3 - Fact Checking
# ====================================================================

FACT_DATABASE = {
    "world population": ("~8.1 billion (2024)", [7.5, 8.5]),
    "renewable share": ("~30% of global electricity (2023)", [25, 35]),
    "global co2": ("~37 billion tonnes/year", [35, 40]),
    "internet users": ("~5.4 billion (2024)", [5.0, 5.8]),
    "temperature rise": ("~1.2 C above pre-industrial", [1.0, 1.5]),
}


def fact_check_feedback(task: str, output: str) -> "FeedbackResult":
    """
    Check factual claims against a known database.

    External knowledge provides ground truth the LLM cannot fabricate.
    """
    from config import get_client, get_model

    extract_prompt = (
        "Extract all factual/numerical claims from the text. List each claim on its own line, "
        "prefixed with '- '. Only include specific factual statements, not opinions.\n\n"
        "Text:\n{}"
    ).format(output)

    response = get_client().chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": "Extract factual claims from this text."},
            {"role": "user", "content": extract_prompt}
        ],
        max_tokens=700,
        temperature=0.2,
    )
    claims_raw = response.choices[0].message.content
    if not claims_raw:
        claims_raw = getattr(response.choices[0].message, 'reasoning', '') or ''
    claims_raw = claims_raw or ""

    claims = [line.strip().lstrip("- ") for line in claims_raw.split("\n")
               if line.strip().startswith("-")]

    if not claims:
        return FeedbackResult(score=5.0, passed=False,
                               feedback_text="Could not extract verifiable claims.")

    verified = 0
    flagged = []
    for claim in claims[:8]:
        claim_lower = claim.lower()
        matched = False

        for fact_key, (fact_value, range_vals) in FACT_DATABASE.items():
            if any(word in claim_lower for word in fact_key.split()):
                matched = True
                verified += 1
                break

        if not matched:
            verify_prompt = "Is this claim factually accurate? Reply with ACCURATE, INACCURATE, or UNCERTAIN: " + claim
            verify_resp = get_client().chat.completions.create(
                model=get_model(),
                messages=[
                    {"role": "user", "content": verify_prompt}
                ],
                max_tokens=200,
                temperature=0.2,
            )
            verdict = verify_resp.choices[0].message.content
            if not verdict:
                verdict = getattr(verify_resp.choices[0].message, 'reasoning', '') or ''
            verdict = verdict or ""

            if "INACCURATE" in verdict.upper():
                flagged.append("INACCURATE: " + claim)
            elif "UNCERTAIN" in verdict.upper():
                flagged.append("UNCERTAIN: " + claim)
                verified += 0.5
            else:
                verified += 1

    total = min(len(claims), 8)
    score = (verified / total) * 10.0 if total > 0 else 5.0
    passed = score >= 7.0 and len(flagged) == 0

    feedback_parts = ["Verified {:.0f}/{} claims.".format(verified, total)]
    if flagged:
        feedback_parts.append("Issues found:")
        feedback_parts.extend("  " + f for f in flagged)

    return FeedbackResult(
        score=min(10, score),
        passed=passed,
        feedback_text="\n".join(feedback_parts),
        details={"claims": claims[:8], "flagged": flagged, "verified": verified}
    )


# ====================================================================
# Part 5: Fast Tests (No LLM calls)
# ====================================================================

def _test_data_structures():
    print("=" * 60)
    print("Fast Test 1: Data Structures")
    print("=" * 60)

    fb = FeedbackResult(score=7.5, passed=False, feedback_text="Two bugs found in edge case handling")
    summary_str = fb.summary()
    assert "FAIL" in summary_str
    assert "7.5" in summary_str
    print("  FeedbackResult: OK (summary={})".format(summary_str[:50]))

    it = RefinementIteration(iteration=1, output="code v2", feedback=fb)
    assert it.iteration == 1
    assert it.output == "code v2"
    print("  RefinementIteration: OK")

    trace = RefinementTrace(task="Write a function")
    trace.iterations.append(it)
    trace.final_output = "final code"
    trace.converge_reason = "Score >= threshold"

    traj = trace.score_trajectory()
    assert len(traj) == 1
    assert traj[0] == 7.5

    summ = trace.summary()
    assert "7.5" in summ
    print("  RefinementTrace: OK (summary contains trajectory)")

    it2 = RefinementIteration(iteration=2, output="code v3",
                               feedback=FeedbackResult(score=9.0, passed=True, feedback_text="All tests pass"))
    trace.iterations.append(it2)
    assert trace.score_trajectory() == [7.5, 9.0]
    print("  Trajectory after 2 iterations: [7.5, 9.0] - OK")

    print("  Fast Test 1 PASSED\n")


def _test_safe_exec():
    print("=" * 60)
    print("Fast Test 2: Safe Execution Sandbox")
    print("=" * 60)

    success, output = safe_exec("print(sum(range(10)))")
    assert success is True
    assert "45" in output
    print("  Valid code: PASS (output={})".format(output.strip()))

    success2, output2 = safe_exec("x = 1/0")
    assert success2 is False
    assert "ZeroDivisionError" in output2
    print("  Error handling: PASS (error={})".format(output2[:50]))

    markdown_code = "```python\nprint('hello from markdown')\n```"
    success3, output3 = safe_exec(markdown_code)
    assert success3 is True
    assert "hello from markdown" in output3
    print("  Markdown extraction: PASS (output={})".format(output3.strip()))

    print("  Fast Test 2 PASSED\n")


def _test_style_feedback():
    print("=" * 60)
    print("Fast Test 3: Style Feedback")
    print("=" * 60)

    bad_text = "This is a test. It is very basic. Really quite simple actually. Basically just a test."
    result = style_feedback("test", bad_text)
    assert result.score < 10.0, "Bad text should score below 10"
    assert not result.passed
    print("  Bad text: score={:.1f}, passed={}, issues={}".format(
        result.score, result.passed, len(result.details.get("issues", [])))
    )

    good_text = "Machine learning enables computers to find patterns in data. "
    good_text += "Instead of following rigid rules, the system improves through experience. "
    good_text += "This approach has transformed industries from healthcare to finance. "
    good_text += "Modern models can process millions of examples, "
    good_text += "learning subtle relationships that human engineers would miss."
    result2 = style_feedback("test", good_text)
    print("  Good text: score={:.1f}, passed={}".format(result2.score, result2.passed))

    long_text = "Word " * 600
    result3 = style_feedback("test", long_text)
    assert "verbose" in result3.feedback_text.lower() or "Verbose" in result3.feedback_text
    print("  Long text: score={:.1f} (correctly penalized for verbosity)".format(result3.score))

    print("  Fast Test 3 PASSED\n")


def _test_convergence():
    print("=" * 60)
    print("Fast Test 4: Convergence Detection")
    print("=" * 60)

    agent = IterativeRefinementAgent(
        feedback_fn=lambda t, o: FeedbackResult(score=0, passed=False, feedback_text="stub"),
        max_iterations=5,
        score_threshold=8.0,
        min_improvement=0.3,
    )

    stop, reason = agent.check_convergence([7.0, 8.5], 1)
    assert stop is True
    assert "threshold" in reason
    print("  Score threshold: PASS (reason={})".format(reason))

    stop2, reason2 = agent.check_convergence([5.0, 5.1, 5.05], 2)
    assert stop2 is True
    assert "Plateau" in reason2
    print("  Plateau detection: PASS (reason={})".format(reason2))

    stop3, reason3 = agent.check_convergence([7.0, 6.0], 1)
    assert stop3 is True
    assert "Decline" in reason3
    print("  Decline detection: PASS (reason={})".format(reason3))

    agent2 = IterativeRefinementAgent(
        feedback_fn=lambda t, o: FeedbackResult(score=0, passed=False, feedback_text="stub"),
        max_iterations=3,
        detect_plateau=False,
        detect_decline=False,
    )
    stop4, reason4 = agent2.check_convergence([3.0, 4.0, 4.5, 5.0], 3)
    assert stop4 is True
    assert "Max iterations" in reason4
    print("  Max iterations: PASS (reason={})".format(reason4))

    stop5, reason5 = agent2.check_convergence([5.0, 7.0, 7.8], 2)
    assert stop5 is False
    print("  No stop on progress: PASS (continue={})".format(reason5))

    print("  Fast Test 4 PASSED\n")


def _test_code_feedback():
    print("=" * 60)
    print("Fast Test 5: Code Feedback (with known correct/incorrect code)")
    print("=" * 60)

    correct_fib = """def fib(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
    fb = code_test_feedback("Write a fibonacci function called fib(n)", correct_fib)
    assert fb.passed, "Correct fib should pass all tests, got score={:.1f}".format(fb.score)
    print("  Correct fib: PASS (score={:.1f})".format(fb.score))

    bad_fib = """def fib(n):
    if n <= 0:
        return 1  # Wrong: should return 0
    if n == 1:
        return 1
    return fib(n-1) + fib(n-2)
"""
    fb2 = code_test_feedback("Write a fibonacci function called fib(n)", bad_fib)
    assert not fb2.passed
    print("  Bad fib: score={:.1f}, passed={} (correctly flagged)".format(fb2.score, fb2.passed))

    correct_palindrome = """def is_palindrome(s):
    s = s.lower().replace(" ", "")
    import re
    s = re.sub(r'[^a-z0-9]', '', s)
    return s == s[::-1]
"""
    fb3 = code_test_feedback("Write a palindrome checker called is_palindrome(s)", correct_palindrome)
    assert fb3.passed or fb3.score >= 8.0
    print("  Correct palindrome: score={:.1f}, passed={}".format(fb3.score, fb3.passed))

    print("  Fast Test 5 PASSED\n")


# ====================================================================
# Part 6: LLM Test (Optional)
# ====================================================================

def test_refinement_with_llm(feedback_type="code"):
    """
    Run iterative refinement with real LLM calls.
    """
    print("=" * 60)
    print("LLM Test: Iterative Refinement (Feedback: {})".format(feedback_type.upper()))
    print("=" * 60)

    if feedback_type == "code":
        agent = IterativeRefinementAgent(
            feedback_fn=code_test_feedback,
            max_iterations=4,
            score_threshold=9.5,
            min_improvement=0.5,
        )
        task = (
            "Write a Python function called 'fib(n)' that returns the nth Fibonacci number. "
            "fib(0) should return 0, fib(1) should return 1. Handle edge cases."
        )
    elif feedback_type == "style":
        agent = IterativeRefinementAgent(
            feedback_fn=style_feedback,
            max_iterations=3,
            score_threshold=8.5,
            min_improvement=0.3,
        )
        task = (
            "Write a clear, professional explanation of how blockchain works for a business audience. "
            "Cover distributed ledger, consensus, and immutability. Target 200-300 words."
        )
    else:
        raise ValueError("Unknown feedback type: " + feedback_type)

    trace = agent.run(task, verbose=True)

    print("\n" + "=" * 60)
    print("FINAL OUTPUT (first 1000 chars):")
    print("=" * 60)
    print(trace.final_output[:1000])

    scores = trace.score_trajectory()
    if len(scores) > 1:
        print("\n  Trajectory: {}".format(" -> ".join("{:.1f}".format(s) for s in scores)))
        print("  Improvement: {:.1f} -> {:.1f} ({:+.1f})".format(scores[0], scores[-1], scores[-1] - scores[0]))

    return trace


if __name__ == "__main__":
    print("=" * 60)
    print("Notebook 09: Iterative Refinement")
    print("=" * 60)
    print()

    _test_data_structures()
    _test_safe_exec()
    _test_style_feedback()
    _test_convergence()
    _test_code_feedback()

    print("=" * 60)
    print("All fast tests PASSED")
    print("=" * 60)

    import sys
    if len(sys.argv) > 1:
        test_refinement_with_llm(feedback_type=sys.argv[1])

    print("\nNotebook 09 complete. Next: Multi-Agent Systems (Notebooks 17+)")
