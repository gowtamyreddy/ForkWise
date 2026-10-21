"""Evaluate retrieval quality against a golden JSONL file.

Golden entries look like:
  {"question": "...", "expected_doc_ids": ["2005.11401"], "type": "factual"}

Metrics:
- hit@k: share of questions where a chunk from an expected doc is in the top k.
- MRR: mean reciprocal rank of the first chunk from an expected doc (0 if none
  in the top k).
- citation precision: share of the returned top-5 chunks that come from an
  expected doc. This measures whether the citations we would show are
  pointing at the right document.

Unanswerable and ambiguous questions are reported separately, because for
them the correct behavior is insufficient_evidence, not retrieval.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import embed, index as vector_index, ingest, rerank as rerank_module, retrieve  # noqa: E402

EVAL_TOP_K = 5
RERANK_CANDIDATES = 20  # first-stage depth the cross-encoder reorders


def load_golden(path):
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Accept both "expected_doc_id" (a single string) and
            # "expected_doc_ids" (a list); normalize to a list.
            if "expected_doc_ids" not in entry and "expected_doc_id" in entry:
                doc_id = entry["expected_doc_id"]
                entry["expected_doc_ids"] = [doc_id] if doc_id else []
            entries.append(entry)
    return entries


def load_retriever(artifacts_dir):
    artifacts_dir = Path(artifacts_dir)
    chunks = ingest.load_chunks(artifacts_dir / "chunks.jsonl")
    faiss_index = vector_index.load_index(artifacts_dir / "index.faiss")
    embedder = embed.Embedder()
    return retrieve.Retriever(chunks, embedder, faiss_index)


def rank_of_expected(ranked_chunks, expected_doc_ids):
    """Position (1-based) of the first chunk from an expected doc, or None."""
    for position, (chunk, _score) in enumerate(ranked_chunks, start=1):
        if chunk["doc_id"] in expected_doc_ids:
            return position
    return None


def evaluate(retriever, golden, top_k=EVAL_TOP_K, reranker=None):
    """Run every golden question and aggregate the metrics.

    With a reranker, the first stage retrieves RERANK_CANDIDATES chunks and
    the cross-encoder reorders them before the top_k window is taken — the
    same two-stage pipeline the API runs.
    """
    answerable = [
        entry for entry in golden
        if entry.get("type") not in ("unanswerable", "ambiguous")
    ]
    results = []
    for entry in answerable:
        started = time.perf_counter()
        if reranker is not None:
            fused = retriever.hybrid_search(entry["question"], top_k=RERANK_CANDIDATES)
            ranked = rerank_module.rerank(entry["question"], fused, reranker, top_k)
        else:
            ranked = retriever.hybrid_search(entry["question"], top_k=top_k)
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        rank = rank_of_expected(ranked, entry["expected_doc_ids"])
        hits = sum(1 for chunk, _ in ranked if chunk["doc_id"] in entry["expected_doc_ids"])
        results.append(
            {
                "question": entry["question"],
                "type": entry.get("type", "factual"),
                "rank": rank,
                "precision": hits / top_k,
                "retrieval_ms": retrieval_ms,
            }
        )
    return results


def summarize(results):
    total = len(results)
    summary = {"n": total}
    for k in (1, 3, 5):
        summary[f"hit@{k}"] = sum(1 for r in results if r["rank"] is not None and r["rank"] <= k) / total
    summary["mrr"] = sum(1.0 / r["rank"] for r in results if r["rank"] is not None) / total
    summary["citation_precision"] = sum(r["precision"] for r in results) / total
    latencies = sorted(r["retrieval_ms"] for r in results if "retrieval_ms" in r)
    if latencies:
        summary["mean_retrieval_ms"] = int(sum(latencies) / len(latencies))
        summary["p50_retrieval_ms"] = latencies[len(latencies) // 2]
    return summary


def print_table(summary):
    print(f"{'metric':<20}{'value':>8}")
    print("-" * 28)
    print(f"{'questions':<20}{summary['n']:>8}")
    for key in ("hit@1", "hit@3", "hit@5", "mrr", "citation_precision"):
        print(f"{key:<20}{summary[key]:>8.3f}")


def main():
    artifacts_dir = PROJECT_ROOT / "data" / "artifacts"
    raw_args = sys.argv[1:]
    use_rerank = "--rerank" in raw_args
    json_path = None
    if "--json" in raw_args:
        json_idx = raw_args.index("--json")
        json_path = Path(raw_args[json_idx + 1])
        del raw_args[json_idx:json_idx + 2]
    args = [a for a in raw_args if a != "--rerank"]
    golden_path = Path(args[0]) if args else PROJECT_ROOT / "data" / "golden_sample.jsonl"
    print("Loading retriever...")
    retriever = load_retriever(artifacts_dir)
    reranker = None
    if use_rerank:
        print("Loading cross-encoder reranker...")
        reranker = rerank_module.Reranker()
    golden = load_golden(golden_path)
    mode = "hybrid + cross-encoder rerank" if use_rerank else "hybrid (dense + BM25 + RRF)"
    print(f"Evaluating {len(golden)} golden questions (mode: {mode})...\n")
    results = evaluate(retriever, golden, reranker=reranker)
    corpus_doc_ids = {chunk["doc_id"] for chunk in retriever.chunks}
    missing = [
        entry for entry in golden
        if entry.get("expected_doc_ids")
        and not corpus_doc_ids.intersection(entry["expected_doc_ids"])
    ]
    if missing:
        print(f"WARNING: {len(missing)} golden question(s) expect documents that are "
              f"not in this corpus; they will score as misses:\n")
        for entry in missing:
            print(f"  [missing doc] {entry['expected_doc_ids']} {entry['question'][:60]}")
        print()
    for entry in results:
        rank = entry["rank"] if entry["rank"] is not None else "miss"
        print(f"[{entry['type']:<10}] rank={rank:<4} {entry['question'][:70]}")
    print()
    summary = summarize(results)
    print_table(summary)
    if "mean_retrieval_ms" in summary:
        print(f"\nretrieval latency: mean {summary['mean_retrieval_ms']} ms, "
              f"p50 {summary['p50_retrieval_ms']} ms (this machine, models already loaded)")
    if json_path is not None:
        payload = {
            "golden_file": str(golden_path),
            "mode": mode,
            "corpus": {
                "chunks": len(retriever.chunks),
                "documents": len({chunk["doc_id"] for chunk in retriever.chunks}),
            },
            "summary": summary,
            "results": results,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote machine-readable results to {json_path}")
    graceful = [e for e in golden if e.get("type") in ("unanswerable", "ambiguous")]
    if graceful:
        print("\nGraceful-failure questions (excluded from metrics above):")
        for entry in graceful:
            print(f"  [{entry['type']:<12}] {entry['question'][:70]}")


if __name__ == "__main__":
    main()
