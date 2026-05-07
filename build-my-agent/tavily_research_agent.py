"""Tavily-based research agent.

End-to-end pipeline: Live web search -> context assembly -> LLM synthesis with citations.
Reuses the existing OpenAI-compatible vLLM client from config.py.
"""

import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from web_search_tools import tavily_search, WebSearchResponse
from config import get_client, get_model


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class ResearchCitation:
    source_title: str
    source_url: str
    excerpt: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_title": self.source_title,
            "source_url": self.source_url,
            "excerpt": self.excerpt,
            "score": self.score,
        }


@dataclass
class ResearchReport:
    question: str
    answer: str
    citations: List[ResearchCitation]
    sources_consulted: int
    search_time: float
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "sources_consulted": self.sources_consulted,
            "search_time": self.search_time,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TavilyResearchAgent:
    """Agent that searches the live web and synthesizes cited answers.

    Pipeline:
        1. Tavily search (returns cleaned content + scores)
        2. Assemble top-N results into a context window
        3. LLM synthesis with strict source-attribution prompt

    Design notes:
        - Uses 'raw_content' (full page text) when available, otherwise 'content'.
        - Caps each source at ~1500 chars to fit context window.
        - Handles the hermes-model quirk where content may be None
          (falls back to reasoning field).
        - Temperature 0.3 for factual consistency.
    """

    def __init__(
        self,
        max_results: int = 5,
        search_depth: str = "basic",
        sources_for_context: int = 5,
        chars_per_source: int = 1500,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ):
        self.max_results = max_results
        self.search_depth = search_depth
        self.sources_for_context = sources_for_context
        self.chars_per_source = chars_per_source
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.client = get_client()
        self.model = get_model()

    def _extract_content(self, result) -> str:
        """Pick the best available text from a Tavily result."""
        # Prefer raw_content (full page) if available, else cleaned snippet
        text = result.raw_content or result.content or ""
        return text.strip()

    def _build_messages(self, question: str, search_resp: WebSearchResponse) -> List[Dict[str, str]]:
        """Construct the LLM prompt with sources as context."""
        context_parts = []
        for i, r in enumerate(search_resp.results[: self.sources_for_context], start=1):
            text = self._extract_content(r)[: self.chars_per_source]
            context_parts.append(
                f"--- Source {i}: {r.title} ---\n"
                f"URL: {r.url}\n"
                f"Relevance Score: {r.score:.3f}\n"
                f"{text}\n"
            )

        context = "\n".join(context_parts)

        system_prompt = (
            "You are a rigorous research assistant. Your job is to answer the user's question "
            "using ONLY the provided search results.\n\n"
            "Rules:\n"
            "1. Cite every claim with [Source: Title](URL) format.\n"
            "2. If sources conflict, acknowledge the disagreement.\n"
            "3. If the sources do not contain enough information to answer, say so explicitly.\n"
            "4. Do not hallucinate facts not present in the sources.\n"
            "5. Be concise but thorough. Prefer direct quotes when precision matters."
        )

        user_prompt = (
            f"Question: {question}\n\n"
            f"Search Results:\n{context}\n\n"
            f"Provide a comprehensive, cited answer based solely on the above sources."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def research(self, question: str, verbose: bool = True) -> ResearchReport:
        """Research a question using live web search + LLM synthesis.

        Args:
            question: The research question.
            verbose: Whether to print progress to stdout.

        Returns:
            ResearchReport with answer and citations.
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"TAVILY RESEARCH AGENT")
            print(f"Question: {question}")
            print(f"{'=' * 70}")

        # Step 1: Search
        if verbose:
            print(f"\n[1/3] Searching Tavily (depth={self.search_depth}, max_results={self.max_results})...")

        search_resp = tavily_search(
            query=question,
            max_results=self.max_results,
            search_depth=self.search_depth,
            include_raw_content=True,
        )

        if verbose:
            print(f"      -> {len(search_resp.results)} results in {search_resp.response_time:.2f}s")
            for i, r in enumerate(search_resp.results[:3], 1):
                print(f"         {i}. [{r.score:.3f}] {r.title}")

        # Step 2: Assemble citations
        citations = []
        for r in search_resp.results[: self.sources_for_context]:
            citations.append(ResearchCitation(
                source_title=r.title,
                source_url=r.url,
                excerpt=(r.content or "")[:200],
                score=r.score,
            ))

        # Step 3: Synthesize
        if verbose:
            print(f"\n[2/3] Synthesizing answer with {self.model}...")

        messages = self._build_messages(question, search_resp)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        message = response.choices[0].message
        # Hermes-model quirk: content may be None, reasoning field holds the text
        answer = message.content or getattr(message, "reasoning", "") or ""

        if verbose:
            print(f"\n[3/3] Complete.")
            print(f"\n{'=' * 70}")
            print("ANSWER")
            print(f"{'=' * 70}")
            print(answer)
            print(f"{'=' * 70}")
            print(f"Sources consulted: {len(search_resp.results)}")
            print(f"Citations used:    {len(citations)}")
            print(f"Search latency:    {search_resp.response_time:.2f}s")

        return ResearchReport(
            question=question,
            answer=answer,
            citations=citations,
            sources_consulted=len(search_resp.results),
            search_time=search_resp.response_time,
            raw_response=search_resp.to_dict(),
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly to validate integration)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = "What are the latest developments in quantum computing in 2026?"
    else:
        question = " ".join(sys.argv[1:])

    agent = TavilyResearchAgent(max_results=5, search_depth="basic")
    report = agent.research(question, verbose=True)

    print("\n\n--- Structured Report ---")
    print("Question:          " + report.question)
    print("Answer preview:    " + report.answer[:300].replace("\n", " ") + "...")
    print("Sources consulted: " + str(report.sources_consulted))
    print("Search time:       {:.2f}s".format(report.search_time))
    print("Citations:")
    for c in report.citations:
        print("  - {} (score: {:.3f})".format(c.source_title, c.score))
