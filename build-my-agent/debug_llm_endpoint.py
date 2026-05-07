"""
Debug script: test the LLM endpoint directly to see what it returns.
"""

import json
import time
from openai import OpenAI
from config import LLM_CONFIG, get_model

# Create client with long timeout for cold start
client = OpenAI(
    base_url=LLM_CONFIG["base_url"],
    api_key=LLM_CONFIG["api_key"],
    timeout=180,
)
model = get_model()

print(f"Endpoint: {LLM_CONFIG['base_url']}")
print(f"Model:    {model}")
print()

# Test 1: Simple chat completion
print("=" * 60)
print("TEST 1: Simple chat completion")
print("=" * 60)
start = time.time()
try:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer in one sentence."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        max_tokens=50,
        temperature=0.3,
    )
    elapsed = time.time() - start
    print(f"Response time: {elapsed:.1f}s")
    print(f"Full response object type: {type(resp)}")
    print(f"Choices: {resp.choices}")
    if resp.choices:
        choice = resp.choices[0]
        print(f"Choice type: {type(choice)}")
        print(f"Message type: {type(choice.message)}")
        print(f"Message content: {repr(choice.message.content)}")
        print(f"Finish reason: {choice.finish_reason}")
    else:
        print("NO CHOICES returned!")
except Exception as e:
    elapsed = time.time() - start
    print(f"ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")

# Test 2: JSON plan generation (what _plan_analysis does)
print()
print("=" * 60)
print("TEST 2: JSON plan generation")
print("=" * 60)
plan_prompt = """You are a data analysis planner. Given a question and dataset info, output a JSON list of analysis steps. Each step has: tool, parameters, reasoning.

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

start = time.time()
try:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": plan_prompt},
            {"role": "user", "content": "Which continent has the highest total GDP?"},
        ],
        max_tokens=800,
        temperature=0.3,
    )
    elapsed = time.time() - start
    print(f"Response time: {elapsed:.1f}s")
    if resp.choices:
        content = resp.choices[0].message.content
        print(f"Content type: {type(content)}")
        print(f"Content (first 500 chars): {repr(content[:500]) if content else 'NONE'}")
        print(f"Finish reason: {resp.choices[0].finish_reason}")

        # Try parsing with our parser
        from parser import parse_response
        parsed = parse_response(content if content else "")
        print(f"Parsed type: {type(parsed)}")
        print(f"Parsed: {json.dumps(parsed, indent=2, default=str)[:500]}")
except Exception as e:
    elapsed = time.time() - start
    print(f"ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Synthesis (what _synthesize does)
print()
print("=" * 60)
print("TEST 3: Synthesis")
print("=" * 60)
synth_prompt = """You are a data analyst. Answer the user's question based on the analysis execution trace. Be specific - cite numbers, names, and comparisons. Format your answer clearly with key findings. If anomalies occurred, mention how they were handled."""

synth_user = """Question: Which continent has the highest total GDP, and what is the average GDP per country for each continent? List the continents ranked by total GDP.

Dataset: 20 rows, columns: ['name', 'population_millions', 'area_km2', 'gdp_billions_usd', 'continent']

Execution Trace:
Step 1 (query): Direct analysis -> 20 rows, 5 cols (full dataset)

Please provide a clear, specific answer."""

start = time.time()
try:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": synth_prompt},
            {"role": "user", "content": synth_user},
        ],
        max_tokens=600,
        temperature=0.4,
    )
    elapsed = time.time() - start
    print(f"Response time: {elapsed:.1f}s")
    if resp.choices:
        content = resp.choices[0].message.content
        print(f"Content type: {type(content)}")
        print(f"Content: {repr(content) if content else 'NONE'}")
        print(f"Finish reason: {resp.choices[0].finish_reason}")
except Exception as e:
    elapsed = time.time() - start
    print(f"ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")

print()
print("Done.")
