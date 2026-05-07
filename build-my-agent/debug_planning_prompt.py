"""
Debug: test what the reasoning model actually returns for the planning prompt.
"""

import json
import time
from openai import OpenAI
from config import LLM_CONFIG, get_model
from parser import parse_response

client = OpenAI(
    base_url=LLM_CONFIG["base_url"],
    api_key=LLM_CONFIG["api_key"],
    timeout=180,
)
model = get_model()

system_prompt = """You are a data analysis planner. Given a question and dataset info, output a JSON list of analysis steps. Each step has: tool, parameters, reasoning.

Available tools:
- read: already done, no parameters needed
- query: filter/sort/group/projection. Parameters: conditions, sort_by, group_by, select, limit
- stats: summarize or correlate. Parameters: columns (str or list), x_column, y_column
- transform: derive/select/rename/pivot. Parameters: new_column, expression, select, mapping, etc.
- synthesize: no parameters, produces final answer

Dataset: 20 rows, columns: ['name', 'population_millions', 'area_km2', 'gdp_billions_usd', 'continent']
Preview rows: [{'name': 'China', 'population_millions': 1412, 'area_km2': 9597000, 'gdp_billions_usd': 17963, 'continent': 'Asia'}, {'name': 'India', 'population_millions': 1408, 'area_km2': 3287000, 'gdp_billions_usd': 3385, 'continent': 'Asia'}]

Past corrections (apply if relevant):
None

IMPORTANT: Return ONLY a JSON array. No markdown, no explanation."""

for max_tok in [4096, 8192]:
    print(f"\n{'='*70}")
    print(f"max_tokens = {max_tok}")
    print(f"{'='*70}")

    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Which continent has the highest total GDP?"},
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
        print(f"  content type: {type(content).__name__}")
        print(f"  content length: {len(content) if content else 0}")
        print(f"  reasoning length: {len(reasoning) if reasoning else 0}")

        if content:
            print(f"\n  --- Content (first 300 chars) ---")
            print(f"  {content[:300]}")
            print(f"\n  --- Content (last 300 chars) ---")
            print(f"  {content[-300:]}")

            # Try parsing
            parsed = parse_response(content)
            print(f"\n  --- parse_response result ---")
            print(f"  Type: {type(parsed).__name__}")
            if isinstance(parsed, list):
                valid_tools = {"read", "query", "stats", "transform", "synthesize"}
                valid_steps = [s for s in parsed if s.get("tool") in valid_tools]
                print(f"  Steps: {len(valid_steps)} valid steps")
                for i, s in enumerate(valid_steps):
                    print(f"    Step {i+1}: {s.get('tool')} - {s.get('reasoning', '')}")
            else:
                print(f"  Parsed: {json.dumps(parsed, indent=2, default=str)[:500]}")

        if reasoning:
            print(f"\n  --- Reasoning (first 300 chars) ---")
            print(f"  {reasoning[:300]}")
            print(f"\n  --- Reasoning (last 300 chars) ---")
            print(f"  {reasoning[-300:]}")

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
