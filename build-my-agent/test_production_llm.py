"""
Production Tools — Real LLM Integration Tests.

Exercises the 10 production tools through the AdvancedToolAgent with real
vLLM (Hermes) calls. Each query is designed to trigger specific tools.
"""

import json
import time
from agent_tool_integration import AdvancedToolAgent
from tool_registry import build_production_registry

# ============================================================================
# Test Queries — each targets specific production tools
# ============================================================================

QUERIES = [
    # 1. Calculator (math expressions)
    {
        "name": "calculator_basic",
        "query": "Use the calculator tool to compute sqrt(144) + 3 * 5. Tell me the result.",
        "expect_contains": ["27"],
    },
    # 2. String utils (text transformation)
    {
        "name": "string_ops",
        "query": "Use the string_utils tool to convert 'hello world from the agent' to UPPERCASE. What is the result?",
        "expect_contains": ["HELLO WORLD FROM THE AGENT"],
    },
    # 3. List ops (sorting, frequencies)
    {
        "name": "list_sort",
        'query': 'Use the list_ops tool to sort this list: [42, 7, 19, 3, 99, 1]. What is the sorted result?',
        "expect_contains": ["1", "3", "7", "19", "42", "99"],
    },
    # 4. Dict ops (key lookup, invert)
    {
        "name": "dict_lookup",
        "query": 'Use the dict_ops tool to get the value of key "name" from this dict: {"name": "Castalia", "version": 13, "status": "active"}. What is the value?',
        "expect_contains": ["Castalia"],
    },
    # 5. Date/time (parse, weekday)
    {
        "name": "date_parse",
        "query": "Use the date_time tool to parse the date '2024-07-20' and tell me what day of the week it was.",
        "expect_contains": ["Saturday"],
    },
    # 6. Text statistics
    {
        "name": "text_stats",
        "query": 'Use the text_stats tool to analyze this text: "The quick brown fox jumps over the lazy dog. How many words are there?". Tell me the word count.',
        "expect_contains": ["14"],
    },
    # 7. Format converter (JSON to markdown table)
    {
        "name": "format_convert",
        'query': 'Use the format_converter tool to convert this JSON to a markdown table: [{"name": "Alice", "role": "Engineer"}, {"name": "Bob", "role": "Designer"}]. Show me the table.',
        "expect_contains": ["Alice", "Bob", "Engineer", "Designer"],
    },
    # 8. Data validator
    {
        "name": "data_validate",
        "query": 'Use the data_validator tool to validate this data {"username": "ab", "age": 200} against these rules {"username": {"required": true, "type": "string", "min_length": 3}, "age": {"required": true, "type": "number", "max": 130}}. What errors does it find?',
        "expect_contains": ["error", "valid"],
    },
    # 9. Advanced math
    {
        "name": "math_stats",
        "query": "Use the math_advanced tool to compute the mean and standard deviation of [10, 20, 30, 40, 50]. Start with the mean.",
        "expect_contains": ["30"],
    },
    # 10. Encoding (base64)
    {
        "name": "encoding",
        "query": 'Use the encoding_tools tool to base64 encode the text "Castalia Notebook 13". What is the result?',
        "expect_contains": ["Q2FzdGFsaWEgTm90ZWJvb2sgMTM="],
    },
]


def run_all_tests():
    """Run all production tool queries against the real LLM."""
    registry = build_production_registry()
    agent = AdvancedToolAgent(registry, max_steps=5)

    print("=" * 80)
    print("PRODUCTION TOOLS — REAL LLM INTEGRATION TESTS")
    print("=" * 80)
    print(f"Tools available: {sorted(registry.tools.keys())}")
    print(f"Total queries: {len(QUERIES)}")
    print("=" * 80)

    results = []
    total_start = time.time()

    for i, tc in enumerate(QUERIES, 1):
        print(f"\n{'-' * 80}")
        print(f"TEST {i}/{len(QUERIES)}: {tc['name']}")
        print(f"  Query: {tc['query'][:100]}...")
        print(f"{'-' * 80}")

        try:
            result = agent.run(tc["query"], verbose=True)
            answer = result["answer"]
            tool_calls = result["tool_calls"]
            elapsed = result["total_time"]

            # Check expectations
            passed = True
            missing = []
            for expected in tc["expect_contains"]:
                if expected.lower() not in answer.lower():
                    passed = False
                    missing.append(expected)

            status = "PASS" if passed else "FAIL"
            results.append({
                "name": tc["name"],
                "status": status,
                "tool_calls": tool_calls,
                "time": elapsed,
                "missing": missing if not passed else None,
            })

            print(f"\n  [{status}] Answer: {answer[:150]}")
            if not passed:
                print(f"  Missing: {missing}")
            print(f"  Tool calls: {tool_calls}, Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"\n  [ERROR] {e}")
            results.append({
                "name": tc["name"],
                "status": "ERROR",
                "tool_calls": 0,
                "time": 0,
                "missing": [str(e)],
            })

    # ── Summary ────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Test':<25} {'Status':<8} {'Tool Calls':<12} {'Time (s)':<10} {'Missing'}")
    print("-" * 80)
    for r in results:
        missing = ", ".join(r.get("missing") or [])
        print(
            f"{r['name']:<25} {r['status']:<8} {r['tool_calls']:<12} "
            f"{r['time']:<10.1f} {missing}"
        )
    print("-" * 80)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Errors: {errors}")
    print(f"Total wall time: {total_elapsed:.1f}s")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_all_tests()
