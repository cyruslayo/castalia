"""Semantic local search — fastembed BGE (ONNX) + FAISS.

Dense retrieval over the Notebook 15 corpus using Qdrant's fastembed,
which loads BGE-small-en-v1.5 via ONNX Runtime (no PyTorch needed).
Complements TF-IDF by capturing semantic relationships beyond keyword overlap.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from typing import List, Dict, Any

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

import faiss
from notebook15_corpus import CORPUS


# ---------------------------------------------------------------------------
# Lazy-loaded fastembed model (cached at module level)
# ---------------------------------------------------------------------------
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("Loading fastembed BGE model (first call)...")
        _embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("  Model loaded: BAAI/bge-small-en-v1.5 via fastembed")
    return _embed_model


class SemanticLocalSearchEngine:
    """Dense semantic search over the Notebook 15 corpus using fastembed + FAISS.

    fastembed loads BGE-small-en-v1.5 via ONNX Runtime — lightweight, no torch.
    FAISS IndexFlatIP computes inner-product search, which equals cosine
    similarity when vectors are L2-normalized.
    """

    def __init__(self):
        self.documents = CORPUS
        self.dim = 384  # bge-small-en-v1.5
        self._faiss_index = None
        self.embeddings = None
        self._is_indexed = False

    def index(self):
        """Embed all documents and build FAISS index."""
        if not HAS_FASTEMBED:
            raise ImportError(
                "fastembed not installed. "
                "Install: pip install fastembed"
            )

        model = _get_embed_model()

        # Prepare texts: title + content for each document
        texts = [f"{d['title']}. {d['content']}" for d in self.documents]

        print(f"Embedding {len(texts)} documents with fastembed BGE...")
        raw_embeddings = list(model.embed(texts))  # generator → list of np arrays

        # Stack into matrix and L2-normalize for cosine similarity
        self.embeddings = np.stack(raw_embeddings).astype("float32")
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        self.embeddings = self.embeddings / norms

        # Build FAISS index (Inner Product = cosine similarity for normalized vectors)
        self._faiss_index = faiss.IndexFlatIP(self.dim)
        self._faiss_index.add(self.embeddings)

        self._is_indexed = True
        print(f"  FAISS index built: {self._faiss_index.ntotal} vectors, {self.dim}D")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search by semantic similarity. Returns top-k results with cosine scores in [0, 1]."""
        if not self._is_indexed:
            self.index()

        model = _get_embed_model()
        q_emb = list(model.embed([query]))[0].astype("float32")

        # L2-normalize query embedding
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores, indices = self._faiss_index.search(q_emb.reshape(1, -1), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                doc = self.documents[idx]
                results.append({
                    "source": "local_semantic",
                    "doc_id": int(idx),
                    "title": doc["title"],
                    "topic": doc["topic"],
                    "score": round(float(score), 4),
                    "content_preview": doc["content"][:200] + "...",
                })
        return results


if __name__ == "__main__":
    engine = SemanticLocalSearchEngine()
    engine.index()
    results = engine.search(
        "how do transformers capture long-range dependencies", top_k=3
    )
    print("\nSemantic Results:")
    for r in results:
        print(f"  [{r['topic']}] {r['title']} (score: {r['score']})")
