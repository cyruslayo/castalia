"""Web search tools using Tavily API.

Integrates with the ToolRegistry from tool_registry.py.
Tavily returns pre-cleaned content and citations, making it ideal
for agentic retrieval pipelines without HTML parsing overhead.
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import requests

# --- Configuration ---
# Prefer explicit env var; fallback to config.py for centralized projects
try:
    from config import get_tavily_api_key, get_tavily_base_url
except ImportError:
    def get_tavily_api_key():
        return os.environ.get("TAVILY_API_KEY", "")

    def get_tavily_base_url():
        return os.environ.get("TAVILY_BASE_URL", "https://api.tavily.com")

DEFAULT_TIMEOUT = 30
MAX_RESULTS_LIMIT = 20


# ---------------------------------------------------------------------------
# Data contracts (match the project's dataclass-heavy style)
# ---------------------------------------------------------------------------

@dataclass
class WebSearchResult:
    """Single search result from Tavily."""
    title: str
    url: str
    content: str                 # Pre-cleaned, relevant text (snippet)
    raw_content: Optional[str] = None  # Full page text if requested
    score: float = 0.0           # Tavily relevance score
    published_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "raw_content": self.raw_content,
            "score": self.score,
            "published_date": self.published_date,
        }


@dataclass
class WebSearchResponse:
    """Structured response from a Tavily search."""
    query: str
    results: List[WebSearchResult]
    answer: Optional[str] = None       # Pre-generated LLM answer if requested
    response_time: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "response_time": self.response_time,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Core Tavily client
# ---------------------------------------------------------------------------

def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = False,
    include_raw_content: bool = True,
    include_images: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> WebSearchResponse:
    """Execute a Tavily web search.

    Args:
        query: The search query string.
        max_results: Number of results to return (1-20).
        search_depth: "basic" (fast, standard snippets) or "advanced"
            (deeper crawl, longer content). Advanced costs more API quota.
        include_answer: If True, Tavily returns a pre-generated answer
            synthesized from results (useful for quick answers, but we
            typically prefer our own LLM synthesis for consistency).
        include_raw_content: If True, includes full parsed page text.
            Essential for research agents that need deep context.
        include_images: If True, includes image URLs in results.
        timeout: HTTP timeout in seconds.

    Returns:
        WebSearchResponse with structured, scored results.

    Raises:
        ValueError: If API key is not configured.
        requests.HTTPError: On 4xx/5xx from Tavily.
    """
    tavily_api_key = get_tavily_api_key()
    tavily_base_url = get_tavily_base_url()

    if not tavily_api_key or tavily_api_key == "YOUR_TAVILY_API_KEY_HERE":
        raise ValueError(
            "TAVILY_API_KEY not configured. Set the TAVILY_API_KEY environment variable "
            "or update build-my-agent/.env with your Tavily API key."
        )

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    max_results = max(1, min(int(max_results), MAX_RESULTS_LIMIT))

    payload = {
        "api_key": tavily_api_key,
        "query": query.strip(),
        "max_results": max_results,
        "search_depth": search_depth,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
        "include_images": include_images,
    }

    start = time.time()
    response = requests.post(
        f"{tavily_base_url}/search",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()

    results = []
    for r in data.get("results", []):
        results.append(WebSearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            raw_content=r.get("raw_content"),
            score=r.get("score", 0.0),
            published_date=r.get("published_date"),
        ))

    # Tavily usually returns sorted, but enforce descending by score
    results.sort(key=lambda x: x.score, reverse=True)

    return WebSearchResponse(
        query=query,
        results=results,
        answer=data.get("answer"),
        response_time=elapsed,
        raw=data,
    )


# ---------------------------------------------------------------------------
# ToolRegistry-compatible wrapper
# ---------------------------------------------------------------------------

def web_search_tool(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> Dict[str, Any]:
    """ToolRegistry-compatible wrapper for Tavily search.

    Returns a dict with 'success' and 'result'/'error' keys so the
    registry can detect failures and feed structured error messages
    back to the LLM for self-correction (pattern from Notebook 13).
    """
    try:
        resp = tavily_search(
            query=query,
            max_results=min(int(max_results), 5),
            search_depth=search_depth,
            include_raw_content=False,  # Only cleaned snippets for ReAct context
        )
        # Compact results to prevent LLM context overflow
        compact = {
            "query": resp.query,
            "response_time": resp.response_time,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": (r.content or "")[:500],
                    "score": r.score,
                }
                for r in resp.results[:5]
            ],
        }
        return {
            "success": True,
            "result": compact,
            "query": query,
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": (
                f"Tavily search timed out after {DEFAULT_TIMEOUT}s. "
                "Try a simpler query or reduce max_results."
            ),
            "query": query,
        }
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "unknown"
        text = e.response.text[:200] if e.response and e.response.text else "no details"
        return {
            "success": False,
            "error": f"Tavily API error {status}: {text}",
            "query": query,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Search failed: {type(e).__name__}: {str(e)}",
            "query": query,
        }


# ---------------------------------------------------------------------------
# Registry registration helper
# ---------------------------------------------------------------------------

def register_tavily_tools(registry) -> None:
    """Register Tavily search tools with a ToolRegistry instance.

    Usage:
        from tool_registry import ToolRegistry
        from web_search_tools import register_tavily_tools

        registry = ToolRegistry()
        register_tavily_tools(registry)
    """
    # Lazy import to avoid circular dependencies at module load time
    from tool_registry import ToolDefinition, ParameterSchema

    search_def = ToolDefinition(
        name="web_search",
        description=(
            "Search the live internet for current, factual information. "
            "Returns structured results with title, URL, cleaned content snippet, "
            "full page text (raw_content), and a relevance score. "
            "Use this for recent events, current facts, or topics outside "
            "your training knowledge cutoff. Always cite sources using the provided URLs. "
            "Prefer 'basic' depth for quick lookups; 'advanced' for deep research."
        ),
        parameters=[
            ParameterSchema(
                name="query",
                type="string",
                description=(
                    "The search query. Be specific. Include dates or names for precision. "
                    "Example: 'OpenAI GPT-5 release date 2025' rather than 'GPT-5'."
                ),
                required=True,
            ),
            ParameterSchema(
                name="max_results",
                type="integer",
                description="Number of results to retrieve (1-20). Default 5.",
                required=False,
                default=5,
            ),
            ParameterSchema(
                name="search_depth",
                type="string",
                description=(
                    "Search depth: 'basic' (fast, standard snippets) or "
                    "'advanced' (deeper crawl, more content). Default 'basic'."
                ),
                required=False,
                default="basic",
                enum=["basic", "advanced"],
            ),
        ],
        return_type="dict",
        return_description="Structured search response with results, scores, and metadata",
        examples=[
            {"input": {"query": "latest AI model releases May 2026", "max_results": 5}},
            {"input": {"query": "Tesla FSD v13 safety statistics", "search_depth": "advanced"}},
        ],
    )

    registry.register(search_def, web_search_tool)
