"""Debug: test basic LLM connectivity."""

from config import get_client, get_model

client = get_client()
model = get_model()

print("Client:", type(client))
print("Model:", model)

# Test 1: Simple echo
print("\n" + "=" * 60)
print("Test 1: Simple echo test")
print("=" * 60)

try:
    response1 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Say hello in one word."},
        ],
        max_tokens=100,
        temperature=0.3,
    )
    content1 = response1.choices[0].message.content
    print("Response 1:", content1)
except Exception as e:
    print("Error 1:", type(e).__name__, str(e))

# Test 2: JSON output
print("\n" + "=" * 60)
print("Test 2: JSON output test")
print("=" * 60)

try:
    prompt2 = (
        "Evaluate this answer and return a JSON object with 'score' (1-10) and 'feedback' (string).\n\n"
        "Answer to evaluate: 'The sky is blue and grass is green.'\n\n"
        "Return ONLY a JSON object, no other text."
    )
    response2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt2},
        ],
        max_tokens=500,
        temperature=0.3,
    )
    content2 = response2.choices[0].message.content
    print("Response 2:", content2)
except Exception as e:
    print("Error 2:", type(e).__name__, str(e))

# Test 3: The actual critic prompt (simplified)
print("\n" + "=" * 60)
print("Test 3: Simplified critic prompt")
print("=" * 60)

try:
    prompt3 = (
        "You are a quality reviewer.\n\n"
        "Goal: Write a 3-section report about speed of light, capital of France, and light-seconds from Sun to Earth.\n\n"
        "Answer: The speed of light is 300,000 km/s. The capital of France is Paris.\n\n"
        "Score 1-10 and explain. Return as a short text, not JSON."
    )
    response3 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt3},
        ],
        max_tokens=1000,
        temperature=0.3,
    )
    content3 = response3.choices[0].message.content
    print("Response 3 (first 500 chars):", str(content3)[:500] if content3 else "None")
except Exception as e:
    print("Error 3:", type(e).__name__, str(e))

print("\n" + "=" * 60)
print("Debug complete")
print("=" * 60)