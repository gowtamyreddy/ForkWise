"""Build the retrieval artifacts: chunks, embeddings, FAISS index.

Reads data/papers/*.pdf, chunks each paper, embeds every chunk, and writes
data/artifacts/{chunks.jsonl, embeddings.npy, index.faiss}. Run once after
fetching the corpus, and again any time the corpus changes.

Usage: python scripts/build_index.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import embed, index as vector_index, ingest  # noqa: E402

PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PAPERS_DIR}. Run scripts/fetch_corpus.py first.")
        sys.exit(1)

    print(f"Ingesting {len(pdfs)} PDFs...")
    started = time.perf_counter()
    all_chunks = []
    for pdf in pdfs:
        doc_id = pdf.stem
        pages = ingest.read_pdf(pdf, doc_id)
        chunks = ingest.chunk_pages(doc_id, pages)
        all_chunks.extend(chunks)
        print(f"  {doc_id}: {len(pages)} pages -> {len(chunks)} chunks")
    ingest_time = time.perf_counter() - started
    print(f"Ingest: {len(all_chunks)} chunks in {ingest_time:.1f}s")

    print("Embedding chunks...")
    started = time.perf_counter()
    embedder = embed.Embedder()
    vectors = embedder.embed_passages([chunk["text"] for chunk in all_chunks])
    embed_time = time.perf_counter() - started
    print(f"Embed: {vectors.shape} in {embed_time:.1f}s")

    print("Building FAISS index...")
    started = time.perf_counter()
    faiss_index = vector_index.build_index(vectors)
    index_time = time.perf_counter() - started

    ingest.save_chunks(all_chunks, ARTIFACTS_DIR / "chunks.jsonl")
    embedder.save_embeddings(vectors, ARTIFACTS_DIR / "embeddings.npy")
    vector_index.save_index(faiss_index, ARTIFACTS_DIR / "index.faiss")
    print(f"Index built in {index_time:.1f}s; artifacts in {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
