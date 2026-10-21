"""Extractive answering: the answer is a quote, not a generation.

Pipeline: take the top retrieved chunks, split each into sentences, embed
every sentence, and score each sentence against the query by cosine
similarity. The best 1 to 3 sentences above the threshold become the answer,
each with a citation pointing at its exact span in the source document.

Why extractive: no LLM means no hallucinated wording and no API key. The
tradeoff is honest and worth stating in an interview: answers are limited to
what is written verbatim, so synthesis across sentences is minimal and
yes/no questions get a quote rather than a clean "yes".

If the best sentence scores below the threshold, we return
"insufficient_evidence" instead of guessing. That failure mode is a feature:
it is how the system stays honest on questions the corpus cannot answer.
"""

import re

import numpy as np

DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_SPANS = 3
MIN_SENTENCE_WORDS = 4

# How the default threshold was chosen: on the demo corpus, answerable
# questions score around 0.65-0.80, vague questions around 0.40-0.45, and
# off-topic questions below 0.15. 0.5 sits in the gap. Recalibrate if the
# corpus or the embedding model changes; the tests pin this behavior.


def split_sentences(text):
    """Split on sentence-ending punctuation. Drops fragments under 4 words."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if len(part.split()) >= MIN_SENTENCE_WORDS]


def _sentences_with_offsets(text):
    """Yield (sentence, start, end) with offsets into the chunk text."""
    results = []
    cursor = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)
        if start == -1:
            continue
        end = start + len(sentence)
        results.append((sentence, start, end))
        cursor = end
    return results


def extractive_answer(query, ranked_chunks, embedder, threshold=DEFAULT_THRESHOLD,
                      max_spans=DEFAULT_MAX_SPANS):
    """Answer a query with quoted spans from the ranked chunks.

    Returns a dict with status "answered" or "insufficient_evidence".
    """
    candidates = []
    for chunk, _retrieval_score in ranked_chunks:
        for sentence, start, end in _sentences_with_offsets(chunk["text"]):
            candidates.append((sentence, chunk, start, end))

    if not candidates:
        return {"status": "insufficient_evidence", "spans": [], "best_score": 0.0,
                "note": "No sentences found in the retrieved chunks."}

    query_vector = embedder.embed_query(query)
    sentence_texts = [text for text, _, _, _ in candidates]
    sentence_vectors = embedder.embed_passages(sentence_texts)
    scores = sentence_vectors @ query_vector

    order = np.argsort(-scores)
    best_score = float(scores[order[0]])

    if best_score < threshold:
        nearest = [
            {"doc_id": chunk["doc_id"], "page": chunk["page"]}
            for chunk, _ in ranked_chunks[:3]
        ]
        return {
            "status": "insufficient_evidence",
            "spans": [],
            "best_score": round(best_score, 4),
            "note": "No sentence was similar enough to the question to quote.",
            "nearest_chunks": nearest,
        }

    spans = []
    for position in order[:max_spans]:
        score = float(scores[position])
        if score < threshold:
            break
        text, chunk, start, end = candidates[position]
        spans.append(
            {
                "text": text,
                "chunk_text": chunk["text"],
                "rel_start": start,
                "rel_end": end,
                "citation": {
                    "doc_id": chunk["doc_id"],
                    "page": chunk["page"],
                    "char_start": chunk["char_start"] + start,
                    "char_end": chunk["char_start"] + end,
                    "score": round(score, 4),
                },
            }
        )
    return {"status": "answered", "spans": spans, "best_score": round(best_score, 4)}
