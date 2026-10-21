"""Embedding wrapper around sentence-transformers.

All vectors are L2-normalized on the way out. With normalized vectors, the
inner product equals cosine similarity, which is what both FAISS search and
the sentence scoring in answer.py rely on. Keeping normalization in one place
avoids a classic bug: forgetting to normalize the query but not the passages.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _normalize(vectors):
    """L2-normalize each row of a 2D array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def _sanitize(text):
    """Make a string safe for the tokenizer: lone UTF-16 surrogates
    (which some PDF extractors emit) cannot be UTF-8 encoded and crash
    the fast tokenizer. Ingested text is cleaned in ingest.clean_text;
    this is the same guard at the embedding boundary for text that
    arrives by other paths (queries, transcripts, fetched pages)."""
    return text.encode("utf-8", "replace").decode("utf-8")


class Embedder:
    """Loads the model once, embeds passages and queries on demand."""

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts):
        """Embed a list of passages. Returns a normalized (n, dim) array."""
        texts = [_sanitize(text) for text in texts]
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text):
        """Embed one query. Returns a normalized (dim,) vector."""
        vectors = self.model.encode([_sanitize(text)], convert_to_numpy=True, show_progress_bar=False)
        return _normalize(np.asarray(vectors, dtype=np.float32))[0]

    def save_embeddings(self, vectors, path):
        np.save(str(path), vectors)

    def load_embeddings(self, path):
        return np.load(str(path))
