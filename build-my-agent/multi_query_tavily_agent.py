"""Multi-Query Tavily Search Agent.

When the initial query retrieves weak results (score below threshold),
the agent uses the LLM to generate alternative phrasings, searches each,
and merges/deduplicates by URL.

Adapts the Multi-Query Search pattern from Notebook 15 to use live
Tavily search instead of a local FAISS corpus.
"""

import sys
import time
from typing import List, Dict
from dataclasses import dataclass, field

sys.path.insert(0, ".")

from web_search_tools import tavily_search, WebSearchResult
from config import get_client, get_model


@dataclass
class MergedSearchResult:
    """A search result aggregated across multiple query variants."""
    title: str
    url: str
    content: str
    raw_content: str
    score: float
    sources: List[str] = field(default_factory=list)  # which queries found this


class MultiQueryTavilyAgent:
    """Search with automatic LLM-based query reformulation.

    Pipeline:
        1. Run initial Tavily search
        2. If top score < threshold, generate N alternative queries via LLM
        3. Run Tavily for each alternative
        4. Merge results by URL, keep highest score
        5. Return unified ranked list

    Design notes:
        - Uses URL as deduplication key (same page found by different queries)
        - Keeps highest score across all queries that found the same URL
        - Tracks which queries contributed to each result for debugging
    """

    def __init__(
        self,
        score_threshold: float = 0.80,
        max_alternatives: int = 3,
        max_results_per_query: int = 5,
    ):
        self.score_threshold = score_threshold
        self.max_alternatives = max_alternatives
        self.max_results_per_query = max_results_per_query
        self.client = get_client()
        self.model = get_model()

    def _generate_alternatives(self, query: str) -> List[str]:
        """Use LLM to generate rephrased search queries.

        The prompt asks for simple one-line queries with no explanation
        so parsing is robust even if the LLM adds reasoning prefixes.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate " + str(self.max_alternatives) + " alternative search queries for the given question. "
                    "Each should approach the topic from a different angle or use different keywords. "
                    "Return ONLY the queries, one per line, no numbering or explanation."
                ),
            },
            {"role": "user", "content": query},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        msg = response.choices[0].message
        text = msg.content or getattr(msg, "reasoning", "") or ""

        # Parse lines, strip numbering/bullets, remove empty lines
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        cleaned = []
        for line in lines:
            # Strip common prefixes: "1.", "-", "*", etc.
            stripped = line.lstrip("0123456789.-*) ").strip()
            if stripped:
                cleaned.append(stripped)

        return cleaned[: self.max_alternatives]

    def search(self, query: str, verbose: bool = True) -> List[WebSearchResult]:
        """Search with automatic query reformulation when results are poor.

        Args:
            query: The initial search query.
            verbose: Whether to print progress to stdout.

        Returns:
            List of WebSearchResult, sorted by score descending.
        """
        if verbose:
            print("\n" + "=" * 60, flush=True)
            print("MULTI-QUERY TAVILY SEARCH", flush=True)
            print("Query: " + query, flush=True)
            print("=" * 60, flush=True)

        # ------------------------------------------------------------------
        # Step 1: Initial search
        # ------------------------------------------------------------------
        t0 = time.time()
        initial = tavily_search(
            query=query,
            max_results=self.max_results_per_query,
            search_depth="basic",
        )
        top_score = initial.results[0].score if initial.results else 0.0

        if verbose:
            print("\nInitial search: " + str(len(initial.results)) + " results", flush=True)
            print(
                "Top score: {:.3f} (threshold: {:.3f})".format(top_score, self.score_threshold),
                flush=True,
            )
            for i, r in enumerate(initial.results[:3], 1):
                print("  {}. [{:.3f}] {}".format(i, r.score, r.title), flush=True)

        # If initial results are strong enough, return immediately
        if top_score >= self.score_threshold:
            if verbose:
                print("\nScore above threshold -- no reformulation needed.", flush=True)
            return initial.results

        # ------------------------------------------------------------------
        # Step 2: Generate and search alternatives
        # ------------------------------------------------------------------
        if verbose:
            print("\nScore below threshold -- generating alternative queries...", flush=True)

        alternatives = self._generate_alternatives(query)
        if verbose:
            print("Alternatives generated:", flush=True)
            for alt in alternatives:
                print("  -> " + alt, flush=True)

        # Merge results: dict keyed by URL -> WebSearchResult (keep highest score)
        merged: Dict[str, WebSearchResult] = {}

        # Seed with initial results
        for r in initial.results:
            merged[r.url] = r

        # Search each alternative and merge
        for alt in alternatives:
            if verbose:
                print("\nSearching: " + alt, flush=True)
            alt_resp = tavily_search(
                query=alt,
                max_results=self.max_results_per_query,
                search_depth="basic",
            )
            if verbose:
                print("  -> " + str(len(alt_resp.results)) + " results", flush=True)

            for r in alt_resp.results:
                if r.url in merged:
                    # Same page found by a different query -- keep higher score
                    if r.score > merged[r.url].score:
                        merged[r.url] = r
                else:
                    merged[r.url] = r

        # Sort by score descending
        results = sorted(merged.values(), key=lambda x: x.score, reverse=True)

        if verbose:
            elapsed = time.time() - t0
            print("\n" + "=" * 60, flush=True)
            print("MERGED RESULTS", flush=True)
            print("=" * 60, flush=True)
            print("Unique results: " + str(len(results)), flush=True)
            print("Total time:     {:.1f}s".format(elapsed), flush=True)
            for i, r in enumerate(results[:5], 1):
                print("  {}. [{:.3f}] {}".format(i, r.score, r.title), flush=True)

        return results


def main():
    agent = MultiQueryTavilyAgent(
        score_threshold=0.85,  # High bar -- vague queries should trigger reformulation
        max_alternatives=3,
        max_results_per_query=5,
    )

    # Query that is intentionally vague to force reformulation
    query = "recent stuff about artificial intelligence companies"

    results = agent.search(query, verbose=True)

    print("\n" + "=" * 60, flush=True)
    print("TOP 3 FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    for i, r in enumerate(results[:3], 1):
        print(str(i) + ". " + r.title, flush=True)
        print("   " + r.url, flush=True)
        print("   " + r.content[:120] + "...", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
