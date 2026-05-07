"""
Knowledge Base - A dictionary of facts the agent can retrieve.

This simulates a mini database. In production, this would be a vector store
(like Pinecone or Weaviate), but the concept is the same:
  - Store facts with identifiable keys
  - Provide a search function to retrieve them
  - The agent calls the search function as a tool

The knowledge base gives the agent access to **external knowledge** that isn't
in the LLM's training data. This is the foundation of RAG (Retrieval-Augmented Generation).
"""

# The knowledge base: 30 facts across 4 domains
KNOWLEDGE_BASE = {
    # === Geography ===
    "capital of France": "Paris is the capital of France. It is located in northern France on the Seine River.",
    "capital of Japan": "Tokyo is the capital of Japan. It is the most populous prefecture in Japan and part of the Kanto region in eastern Honshu island.",
    "capital of Australia": "Canberra is the capital of Australia. It is located in the Australian Capital Territory (ACT).",
    "capital of Brazil": "Brasilia is the capital of Brazil. It was purpose-built in the 1960s and is located in the center-south of the country.",
    "largest ocean on Earth": "The Pacific Ocean is the largest and deepest of Earth's ocean divisions, covering about 165.25 million square kilometers.",
    "longest river in the world": "The Nile River is traditionally considered the longest river in the world, at about 6,650 km. The Amazon is a close second.",
    "highest mountain on Earth": "Mount Everest is the highest mountain above sea level, at 8,849 meters (29,032 feet).",

    # === Science ===
    "chemical symbol for gold": "The chemical symbol for gold is Au (from Latin 'aurum'). It has atomic number 79.",
    "chemical symbol for silver": "The chemical symbol for silver is Ag (from Latin 'argentum'). It has atomic number 47.",
    "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s or 186,000 mi/s).",
    "gravity on the Moon": "The gravity on the Moon is about 1.625 m/s², which is roughly one-sixth (1/6) of Earth's gravity (9.81 m/s²).",
    "water freezing point": "Water freezes at 0°C (32°F) at standard atmospheric pressure (1 atmosphere).",
    "water boiling point": "Water boils at 100°C (212°F) at standard atmospheric pressure (1 atmosphere).",

    # === History ===
    "year WW2 ended": "World War II ended in 1945. Germany surrendered in May 1945 (V-E Day), and Japan surrendered in September 1945 (V-J Day).",
    "year the first moon landing occurred": "The first moon landing occurred in 1969. Apollo 11 landed on July 20, 1969, with Neil Armstrong and Buzz Aldrin.",
    "who invented the telephone": "Alexander Graham Bell is credited with inventing the first practical telephone in 1876.",
    "year the US Declaration of Independence was signed": "The US Declaration of Independence was signed in 1776, specifically on July 4, 1776.",

    # === Technology ===
    "who founded Microsoft": "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
    "year the first iPhone was released": "The first iPhone was released on June 29, 2007, by Apple Inc. under Steve Jobs' leadership.",
    "what is HTML": "HTML (HyperText Markup Language) is the standard markup language for creating web pages. It describes the structure of a webpage using tags and elements.",
    "what is an API": "An API (Application Programming Interface) is a set of rules that allows different software applications to communicate with each other. It defines the methods and data formats programs can use.",

    # === Math ===
    "value of pi": "Pi (π) is approximately 3.14159265359. It is the ratio of a circle's circumference to its diameter and is an irrational number.",
    "value of e (Euler's number)": "Euler's number (e) is approximately 2.71828182846. It is the base of the natural logarithm and arises naturally in compound interest and growth/decay problems.",
}


def get_fact(topic: str) -> str:
    """
    Direct lookup: exact key match.

    In a real system, this would be a database query.
    """
    if topic in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[topic]
    return f"No fact found for exact key: '{topic}'"


def search_kb(query: str, max_results: int = 5) -> str:
    """
    Search the knowledge base using simple keyword matching.

    This is a naive search: for each fact key, check if any word from
    the query appears in the key (case-insensitive).

    In production, this would be:
    - TF-IDF scoring
    - Vector similarity (embeddings + cosine similarity)
    - A full-text search engine (Elasticsearch, etc.)

    Args:
        query: The search terms (e.g., "capital France" or "speed light")
        max_results: How many matching facts to return

    Returns:
        A formatted string with the matching facts
    """
    # Tokenize the query into individual words, filter out very short ones
    query_words = set(query.lower().split())
    query_words = {w for w in query_words if len(w) > 2}  # Skip "of", "the", etc.

    if not query_words:
        return "No meaningful search terms found. Try more specific keywords."

    # Score each fact by how many query words appear in its key
    results = []
    for key, value in KNOWLEDGE_BASE.items():
        key_lower = key.lower()
        # Count how many query words match this key
        score = sum(1 for word in query_words if word in key_lower)
        if score > 0:
            results.append((score, key, value))

    # Sort by score (descending), then by key name for deterministic order
    results.sort(key=lambda x: (-x[0], x[1]))

    # Take top N results
    top_results = results[:max_results]

    if not top_results:
        # Fallback: check if any query word appears in the VALUE text
        top_results = []
        for key, value in KNOWLEDGE_BASE.items():
            value_lower = value.lower()
            score = sum(1 for word in query_words if word in value_lower)
            if score > 0:
                top_results.append((score, key, value))
        top_results.sort(key=lambda x: (-x[0], x[1]))
        top_results = top_results[:max_results]

    if not top_results:
        return f"No results found for query: '{query}'. Try different keywords."

    # Format the results as a readable string
    lines = [f"Found {len(top_results)} result(s) for '{query}':"]
    for score, key, value in top_results:
        lines.append(f"\n--- {key} (relevance: {score}) ---")
        lines.append(value)

    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    print("=== Knowledge Base Test ===\n")

    print(f"Total facts: {len(KNOWLEDGE_BASE)}\n")

    # Test direct lookup
    print("--- Direct Lookup ---")
    print(get_fact("capital of France"))
    print(get_fact("nonexistent key"))

    # Test search
    print("\n--- Search: 'capital France' ---")
    print(search_kb("capital France"))

    print("\n--- Search: 'speed light' ---")
    print(search_kb("speed light"))

    print("\n--- Search: 'water' ---")
    print(search_kb("water"))

    print("\n--- Search: 'capital' (broad) ---")
    print(search_kb("capital"))

    print("\n--- Search: 'moon' (in value, not key) ---")
    print(search_kb("moon"))
