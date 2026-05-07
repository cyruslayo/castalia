"""Quick smoke test for Tavily API key."""
import sys
sys.path.insert(0, ".")

from web_search_tools import tavily_search

try:
    resp = tavily_search("Python programming language latest features", max_results=2)
    print(f"OK SUCCESS: {len(resp.results)} results in {resp.response_time:.2f}s")
    for i, r in enumerate(resp.results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.title}")
        print(f"     {r.url}")
        print(f"     {r.content[:90]}...")
    print("\nKey is valid — ready to integrate.")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
