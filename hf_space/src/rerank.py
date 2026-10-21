"""Cross-encoder reranking: the second stage of the retrieval pipeline.

Why two stages: the first stage (hybrid_search in retrieve.py) optimizes for
recall — it pulls 20 candidates cheaply with dense + BM25 signals, each of
which sees the query and the chunk independently. A cross-encoder reads the
query and a chunk *together* through one transformer, which is far better at
judging relevance, but too slow to run over the whole corpus. Running it over
the fused top-20 costs a fraction of a second and reorders candidates by
joint relevance.

This exists because of the golden-set failure analysis: two of the three
rank-1 misses were retrieval ordering problems (query-term dilution and
broad-vocabulary magnet chunks). RRF fusion ranks those chunks highly because
generic terms match strongly across both signals; the cross-encoder reads the
actual question and demotes chunks that merely share vocabulary.
"""

from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Loads the cross-encoder once; scores (query, chunk) pairs on demand."""

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model = CrossEncoder(model_name)

    def score(self, query, texts):
        """Score each text against the query. Returns a list of floats."""
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        return [float(score) for score in self.model.predict(pairs)]


def rerank(query, candidates, reranker, top_k):
    """Reorder fused candidates by cross-encoder score.

    candidates: [(chunk, fused_score)] from Retriever.hybrid_search.
    Returns up to top_k (chunk, ce_score) pairs, best first.
    """
    scores = reranker.score(query, [chunk["text"] for chunk, _ in candidates])
    rescored = sorted(
        zip(candidates, scores), key=lambda pair: pair[1], reverse=True
    )
    return [(chunk, score) for (chunk, _fused), score in rescored[:top_k]]
