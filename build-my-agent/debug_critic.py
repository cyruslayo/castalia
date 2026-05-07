"""Debug: see what the LLM actually returns for the critic prompt."""

from reflection_agent import build_critic_prompt, _extract_json
from config import get_client, get_model

goal = "Write a 3-section report with bullet points"
answer = "The speed of light is 300,000 km/s. The capital of France is Paris."
threshold = 7.0

prompt = build_critic_prompt(goal, answer, threshold)

print("=" * 60)
print("CRITIC PROMPT (last 500 chars):")
print("=" * 60)
print(prompt[-500:])

print("\n" + "=" * 60)
print("MAKING LLM CALL...")
print("=" * 60)

client = get_client()
response = client.chat.completions.create(
    model=get_model(),
    messages=[{"role": "system", "content": prompt}],
    max_tokens=2048,
    temperature=0.3,
)

raw = response.choices[0].message.content
print("\nRAW LLM RESPONSE (first 2000 chars):")
print("=" * 60)
print(raw[:2000])
print("\n" + "=" * 60)

# Try to extract JSON
parsed = _extract_json(raw)
print("EXTRACTED JSON:")
print(parsed)
