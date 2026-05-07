"""
Debug: test if increasing max_tokens lets the reasoning model produce content.
"""

import time
from openai import OpenAI
from config import LLM_CONFIG, get_model

client = OpenAI(
    base_url=LLM_CONFIG["base_url"],
    api_key=LLM_CONFIG["api_key"],
    timeout=180,
)
model = get_model()

# Test with high token budget
for max_tok in [2048, 4096, 8192]:
    print(f"\n{'='*60}")
    print(f"max_tokens = {max_tok}")
    print(f"{'='*60}")

    try:
        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer in one sentence."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            max_tokens=max_tok,
            temperature=0.3,
        )
        elapsed = time.time() - start
        choice = resp.choices[0]
        content = choice.message.content
        reasoning = getattr(choice.message, 'reasoning', None)
        print(f"  Time: {elapsed:.1f}s")
        print(f"  finish_reason: {choice.finish_reason}")
        print(f"  content: {repr(content)[:200] if content else 'None'}")
        if reasoning:
            print(f"  reasoning (first 150 chars): {repr(reasoning[:150])}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

# Also test max_completion_tokens (OpenAI reasoning API param)
print(f"\n{'='*60}")
print(f"max_completion_tokens = 4096")
print(f"{'='*60}")
try:
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer in one sentence."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        max_completion_tokens=4096,
        temperature=0.3,
    )
    elapsed = time.time() - start
    choice = resp.choices[0]
    content = choice.message.content
    reasoning = getattr(choice.message, 'reasoning', None)
    print(f"  Time: {elapsed:.1f}s")
    print(f"  finish_reason: {choice.finish_reason}")
    print(f"  content: {repr(content)[:200] if content else 'None'}")
    if reasoning:
        print(f"  reasoning (first 150 chars): {repr(reasoning[:150])}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
