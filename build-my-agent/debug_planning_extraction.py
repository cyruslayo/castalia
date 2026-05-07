"""
Debug: test the planning JSON extraction with the actual agent code.
"""

import json
import re
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

question = "Which continent has the highest total GDP, and what is the average GDP per country for each continent? List the continents ranked by total GDP."

print(f"Sending planning request...")
start = time.time()
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ],
    max_tokens=4096,
    temperature=0.3,
    timeout=180,
)
elapsed = time.time() - start
print(f"Response in {elapsed:.1f}s")

choice = resp.choices[0]
content = choice.message.content
reasoning = getattr(choice.message, 'reasoning', None)

print(f"\nfinish_reason: {choice.finish_reason}")
print(f"content type: {type(content).__name__}")
print(f"content length: {len(content) if content else 0}")
print(f"reasoning length: {len(reasoning) if reasoning else 0}")

# Try the exact extraction logic from the agent
valid_tools = {"read", "query", "stats", "transform", "synthesize"}
raw = content

if raw is None:
    raw = reasoning
    print(f"\n[!] Content was None, using reasoning field")

# Step 1: Direct JSON parse
print(f"\n--- Attempting direct json.loads() on raw text ---")
print(f"Raw text (first 100 chars): {repr(raw[:100])}")
try:
    parsed = json.loads(raw)
    print(f"Direct JSON parse SUCCESS: type={type(parsed).__name__}")
    if isinstance(parsed, list):
        steps = [s for s in parsed if s.get("tool") in valid_tools]
        print(f"  Valid steps: {len(steps)}")
        for i, s in enumerate(steps):
            print(f"    Step {i+1}: tool={s.get('tool')}, reasoning={s.get('reasoning', '')[:80]}")
    else:
        print(f"  Not a list: {json.dumps(parsed, indent=2, default=str)[:300]}")
except (json.JSONDecodeError, TypeError) as e:
    print(f"Direct JSON parse FAILED: {e}")

# Step 2: Regex extraction
print(f"\n--- Attempting regex JSON extraction ---")
json_match = re.search(r'\[[\s\S]*\]', raw)
if json_match:
    print(f"Regex found JSON array (length: {len(json_match.group())})")
    try:
        parsed = json.loads(json_match.group())
        print(f"Regex JSON parse SUCCESS: type={type(parsed).__name__}")
        if isinstance(parsed, list):
            steps = [s for s in parsed if s.get("tool") in valid_tools]
            print(f"  Valid steps: {len(steps)}")
            for i, s in enumerate(steps):
                print(f"    Step {i+1}: tool={s.get('tool')}, reasoning={s.get('reasoning', '')[:80]}")
    except json.JSONDecodeError as e:
        print(f"Regex JSON parse FAILED: {e}")
        print(f"  Extracted text (first 300 chars): {json_match.group()[:300]}")
else:
    print(f"Regex found NO JSON array in raw text")

# Step 3: Try parse_response (old approach)
print(f"\n--- Attempting parse_response (old approach) ---")
parsed_old = parse_response(raw)
print(f"parse_response result: type={type(parsed_old).__name__}")
if isinstance(parsed_old, dict):
    print(f"  Keys: {list(parsed_old.keys())}")
    for k, v in parsed_old.items():
        print(f"  {k}: {repr(v)[:200]}")

# Full content for inspection
print(f"\n--- Full content ---")
print(content if content else "(None)")
