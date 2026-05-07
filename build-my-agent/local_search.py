"""Local TF-IDF search engine over the Notebook 15 corpus.

Implements term-frequency x inverse-document-frequency from scratch,
matching the pedagogical approach in Notebook 15.
"""

import math
import string
from collections import Counter
from typing import List, Dict, Any

from notebook15_corpus import CORPUS


class LocalTFIDFSearchEngine:
    """TF-IDF search engine built from scratch — no sklearn or external libraries."""

    def __init__(self):
        self.documents = []
        self.doc_tokens = []
        self.doc_freqs = Counter()
        self.num_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, remove punctuation, split into words."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def index(self):
        """Index the corpus from notebook15_corpus.CORPUS."""
        self.documents = CORPUS
        self.num_docs = len(CORPUS)
        self.doc_tokens = []

        for doc in CORPUS:
            tokens = self._tokenize(doc["title"] + " " + doc["content"])
            self.doc_tokens.append(tokens)

        self.doc_freqs = Counter()
        for tokens in self.doc_tokens:
            unique = set(tokens)
            for term in unique:
                self.doc_freqs[term] += 1

        print(f"Indexed {self.num_docs} documents")
        print(f"Vocabulary size: {len(self.doc_freqs):,} unique terms")

    def _tf(self, term: str, tokens: List[str]) -> float:
        """Term frequency: count / total tokens."""
        return tokens.count(term) / len(tokens) if tokens else 0

    def _idf(self, term: str) -> float:
        """Inverse document frequency: log(N / (1 + df))."""
        df = self.doc_freqs.get(term, 0)
        return math.log(self.num_docs / (1 + df))

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the corpus and return top-k results with TF-IDF scores."""
        query_tokens = self._tokenize(query)
        scores = []
        for idx, doc_tokens in enumerate(self.doc_tokens):
            score = sum(self._tf(t, doc_tokens) * self._idf(t) for t in query_tokens)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                doc = self.documents[idx]
                results.append({
                    "source": "local",
                    "doc_id": idx,
                    "title": doc["title"],
                    "topic": doc["topic"],
                    "score": round(score, 4),
                    "content_preview": doc["content"][:200] + "...",
                })
        return results


if __name__ == "__main__":
    engine = LocalTFIDFSearchEngine()
    engine.index()
    results = engine.search("transformer architecture attention", top_k=3)
    print("\nResults:")
    for r in results:
        print(f"  [{r['topic']}] {r['title']} (score: {r['score']})")
