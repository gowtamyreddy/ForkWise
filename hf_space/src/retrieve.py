"""Hybrid retrieval: dense (FAISS) plus sparse (BM25), fused with RRF.

Why hybrid: dense retrieval understands paraphrase ("how is answer quality
measured" matches "we evaluate generation quality"), while BM25 is exact
about rare terms (paper names, metric names like "MRR"). Each fails where the
other succeeds, so we run both and fuse the rankings.

Why reciprocal rank fusion (RRF): it needs no score normalization between
two incomparable scoring systems (cosine similarity vs BM25 scores). It only
uses ranks: fused_score = sum over systems of 1 / (k + rank), with k = 60.
A chunk that both systems rank highly beats a chunk that only one ranks
highly, which is exactly the behavior we want.
"""

from rank_bm25 import BM25Okapi

from . import index as vector_index

RRF_K = 60


def tokenize(text):
    """Simple whitespace tokenizer, lowercased. Shared by index and query."""
    return text.lower().split()


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Fuse ranked id lists into one ordering. Returns [(id, score)].

    Each ranking is a list of ids, best first. Ids may repeat across rankings;
    scores accumulate, so ids ranked highly by several systems rise to the top.
    """
    fused_scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return ordered


class Retriever:
    """Owns chunks, the FAISS index, the BM25 index, and the embedder."""

    def __init__(self, chunks, embedder, faiss_index):
        self.chunks = chunks
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    def dense_search(self, query, top_k):
        """FAISS cosine search. Returns [(chunk_id, score)], best first."""
        query_vector = self.embedder.embed_query(query)
        return vector_index.search(self.faiss_index, query_vector, top_k)

    def bm25_search(self, query, top_k):
        """BM25 keyword search. Returns [(chunk_id, score)], best first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(chunk_id, float(scores[chunk_id])) for chunk_id in ranked[:top_k]]

    def hybrid_search(self, query, top_k=5, dense_k=20, bm25_k=20):
        """Run both retrievers, fuse with RRF, return [(chunk, fused_score)]."""
        dense_ranking = [chunk_id for chunk_id, _ in self.dense_search(query, dense_k)]
        bm25_ranking = [chunk_id for chunk_id, _ in self.bm25_search(query, bm25_k)]
        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
        return [(self.chunks[chunk_id], score) for chunk_id, score in fused[:top_k]]
