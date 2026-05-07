"""Debug: see the full response structure."""

from config import get_client, get_model

goal = "Write a 3-section report with bullet points"
answer = "The speed of light is 300,000 km/s. The capital of France is Paris."

prompt = (
    "You are a quality reviewer. Your job is to evaluate an answer and provide constructive feedback.\n\n"
    "## Original Goal\n"
    + goal + "\n\n"
    "## Answer to Critique\n"
    + answer + "\n\n"
    "## Your Task\n"
    "Evaluate the answer on 4 dimensions, each scored from 1-10:\n\n"
    "1. Accuracy (1-10): Are the facts correct?\n"
    "2. Completeness (1-10): Does it address ALL parts of the goal?\n"
    "3. Clarity (1-10): Is it well-organized and easy to understand?\n"
    "4. Format (1-10): Does it follow the requested format (if any)?\n\n"
    "For each dimension, provide a score (1-10) and a brief explanation.\n\n"
    "Then, list any specific issues found. For each issue, include:\n"
    "- category: The dimension it affects\n"
    "- issue: What is wrong\n"
    "- suggestion: How to fix it\n\n"
    "Finally, calculate the overall score as the average of the 4 dimension scores.\n"
    "State whether the overall score meets the threshold of 7.0/10.\n\n"
    "## Output Format\n"
    "Return a SINGLE JSON object with this exact structure:\n"
    '{\n'
    '  "score": 7.5,\n'
    '  "scores": {\n'
    '    "accuracy": 8,\n'
    '    "completeness": 6,\n'
    '    "clarity": 9,\n'
    '    "format": 7\n'
    '  },\n'
    '  "critiques": [\n'
    '    {\n'
    '      "category": "completeness",\n'
    '      "issue": "Only 2 of 3 topics covered",\n'
    '      "suggestion": "Add the 3rd topic"\n'
    '    }\n'
    '  ]\n'
    '}'
)

print("=" * 60)
print("MAKING LLM CALL (user message version)...")
print("=" * 60)

client = get_client()
response = client.chat.completions.create(
    model=get_model(),
    messages=[
        {"role": "user", "content": prompt},
    ],
    max_tokens=2048,
    temperature=0.3,
)

raw = response.choices[0].message.content
print("\nRAW LLM RESPONSE (first 2000 chars):")
print("=" * 60)
if raw:
    print(raw[:2000])
else:
    print("LLM returned None/empty content")

print("\nFull response details:")
print("  Choices:", len(response.choices))
if len(response.choices) > 0:
    msg = response.choices[0].message
    print("  Message type:", type(msg))
    print("  Message dict:", msg.__dict__ if hasattr(msg, '__dict__') else "no __dict__")
