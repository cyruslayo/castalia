"""Hybrid Search: Tavily (live web) + Local TF-IDF + Local BGE Semantic.

Three-source fusion:
  1. Tavily web search — current/recent information
  2. Local TF-IDF — keyword-based matching over Notebook 15 corpus
  3. Local BGE Semantic — dense embedding similarity for conceptual matching

All three score sets are normalized to [0, 1] and fused with configurable weights.

Concept from Notebook 15: true hybrid retrieval combines sparse (TF-IDF/keyword)
and dense (semantic/embedding) methods. Here we extend that to also include live
web search, giving the agent access to both foundational knowledge (local corpus)
and current events (web).
"""

import sys
sys.path.insert(0, ".")

from typing import List
from dataclasses import dataclass

from web_search_tools import tavily_search
from local_search import LocalTFIDFSearchEngine
from semantic_local_search import SemanticLocalSearchEngine


@dataclass
class HybridSearchResult:
    source: str          # 'web', 'local_tfidf', or 'local_semantic'
    title: str
    url: str
    content_preview: str
    score: float         # normalized fused score
    raw_score: float     # original score before weighting


class HybridSearchAgent:
    """Three-source hybrid search agent.

    Queries Tavily (web), local TF-IDF, and local BGE semantic simultaneously,
    normalizes all score sets to [0, 1], applies weights, and returns a unified
    ranked list.

    Default weights: web=0.4, tfidf=0.2, semantic=0.4
    Semantic gets high weight because it captures conceptual matches that TF-IDF
    misses; web gets high weight for currency.
    """

    def __init__(
        self,
        web_weight: float = 0.4,
        tfidf_weight: float = 0.2,
        semantic_weight: float = 0.4,
    ):
        self.web_weight = web_weight
        self.tfidf_weight = tfidf_weight
        self.semantic_weight = semantic_weight

        self.local_tfidf = LocalTFIDFSearchEngine()
        self.local_tfidf.index()

        self.local_semantic = SemanticLocalSearchEngine()
        # Semantic indexing is lazy (happens on first search)

    def _normalize(self, scores: List[float]) -> List[float]:
        """Min-max normalize a list of scores to [0, 1]."""
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s > min_s else 1
        return [(s - min_s) / range_s for s in scores]

    def search(
        self, query: str, top_k: int = 5, verbose: bool = True
    ) -> List[HybridSearchResult]:
        """Run three-source hybrid search and return top-k merged results."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"THREE-SOURCE HYBRID SEARCH: {query}")
            print(
                f"  web={self.web_weight}  tfidf={self.tfidf_weight}  "
                f"semantic={self.semantic_weight}"
            )
            print(f"{'='*60}")

        # ------------------------------------------------------------------
        # 1. Web search via Tavily
        # ------------------------------------------------------------------
        web_resp = tavily_search(
            query=query,
            max_results=top_k,
            search_depth="basic",
            include_raw_content=False,
        )
        web_results = []
        for r in web_resp.results:
            web_results.append(
                HybridSearchResult(
                    source="web",
                    title=r.title,
                    url=r.url,
                    content_preview=(r.content or "")[:200],
                    score=r.score * self.web_weight,
                    raw_score=r.score,
                )
            )

        if verbose:
            print(f"\n[Web] {len(web_results)} results from Tavily")
            for i, r in enumerate(web_results[:3], 1):
                print(f"  {i}. [{r.raw_score:.3f}] {r.title}")

        # ------------------------------------------------------------------
        # 2. Local TF-IDF search
        # ------------------------------------------------------------------
        tfidf_raw = self.local_tfidf.search(query, top_k=top_k)
        tfidf_norm = self._normalize([r["score"] for r in tfidf_raw])

        tfidf_results = []
        for r, norm in zip(tfidf_raw, tfidf_norm):
            tfidf_results.append(
                HybridSearchResult(
                    source="local_tfidf",
                    title=r["title"],
                    url=f"[local_tfidf:doc_{r['doc_id']}]",
                    content_preview=r["content_preview"],
                    score=norm * self.tfidf_weight,
                    raw_score=r["score"],
                )
            )

        if verbose:
            print(f"\n[Local TF-IDF] {len(tfidf_results)} results")
            for i, r in enumerate(tfidf_results[:3], 1):
                print(f"  {i}. [{r.raw_score:.3f}→{r.score/self.tfidf_weight:.3f}] {r.title}")

        # ------------------------------------------------------------------
        # 3. Local BGE Semantic search
        # ------------------------------------------------------------------
        semantic_raw = self.local_semantic.search(query, top_k=top_k)
        # Semantic scores from BGE+FAISS (inner product of normalized vectors)
        # are already effectively in [0, 1] as cosine similarity, but we
        # min-max normalize anyway to be safe.
        semantic_norm = self._normalize([r["score"] for r in semantic_raw])

        semantic_results = []
        for r, norm in zip(semantic_raw, semantic_norm):
            semantic_results.append(
                HybridSearchResult(
                    source="local_semantic",
                    title=r["title"],
                    url=f"[local_semantic:doc_{r['doc_id']}]",
                    content_preview=r["content_preview"],
                    score=norm * self.semantic_weight,
                    raw_score=r["score"],
                )
            )

        if verbose:
            print(f"\n[Local Semantic] {len(semantic_results)} results")
            for i, r in enumerate(semantic_results[:3], 1):
                print(f"  {i}. [{r.raw_score:.3f}→{r.score/self.semantic_weight:.3f}] {r.title}")

        # ------------------------------------------------------------------
        # 4. Merge all three sources and rank
        # ------------------------------------------------------------------
        merged = web_results + tfidf_results + semantic_results
        merged.sort(key=lambda x: x.score, reverse=True)

        if verbose:
            print(f"\n[Hybrid] Top {top_k} merged results:")
            for i, r in enumerate(merged[:top_k], 1):
                src_tag = {
                    "web": "[WEB]",
                    "local_tfidf": "[LOC-TF]",
                    "local_semantic": "[LOC-SEM]",
                }.get(r.source, "[?]")
                print(f"  {i}. {src_tag} {r.title}")
                print(f"     fused={r.score:.3f}  raw={r.raw_score:.3f}  ({r.source})")

        return merged[:top_k]


def main():
    agent = HybridSearchAgent(
        web_weight=0.4,
        tfidf_weight=0.2,
        semantic_weight=0.4,
    )

    # Query that stresses SEMANTIC understanding over keyword matching.
    # "read entire sentences at once instead of word by word" is a conceptual
    # description of transformers (parallel self-attention vs sequential RNNs).
    # TF-IDF will struggle because the exact keywords don't match.
    # BGE semantic search will strongly match the transformer document.
    query = (
        "neural networks that read entire sentences at once "
        "instead of word by word"
    )

    results = agent.search(query, top_k=5, verbose=True)

    print(f"\n{'='*60}")
    print("DETAILED TOP 5")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r.source.upper()}] {r.title}")
        print(f"   URL: {r.url}")
        print(f"   Fused score: {r.score:.3f}  (raw: {r.raw_score:.3f})")
        print(f"   {r.content_preview}")


if __name__ == "__main__":
    main()
