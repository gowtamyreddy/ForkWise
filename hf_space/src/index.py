"""FAISS vector index wrapper.

We use IndexFlatIP (exact search, inner product). Because the embedder
returns normalized vectors, inner product equals cosine similarity, so this
is exact cosine nearest-neighbor search. No approximation, no training step,
which keeps the index build deterministic and easy to explain.

The index stores vectors only. The mapping from index position to chunk is
positional: index row i corresponds to chunks[i]. That keeps this module
small; retrieve.py owns the chunk bookkeeping.

Swapping in pgvector later: replace build_index/search/save_index/load_index
with a table-backed equivalent. Nothing else in the codebase touches FAISS
directly.
"""

import faiss
import numpy as np


def build_index(vectors):
    """Build an exact inner-product index from a (n, dim) float array."""
    vectors = np.asarray(vectors, dtype=np.float32)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def save_index(index, path):
    faiss.write_index(index, str(path))


def load_index(path):
    return faiss.read_index(str(path))


def search(index, query_vector, top_k):
    """Return a list of (chunk_position, score) sorted by score, best first."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    scores, ids = index.search(query, top_k)
    return [(int(chunk_id), float(score)) for chunk_id, score in zip(ids[0], scores[0])]
