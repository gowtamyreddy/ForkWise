"""Bridge: EvidenceRAG (portfolio project 1) instrumented by Forkwise.

Run with EvidenceRAG's virtualenv, from this folder:
    ../evidencerag/.venv/bin/python evidencerag_bridge.py
It traces one real EvidenceRAG query (hybrid retrieval -> rerank ->
extractive answer) into demo/demo_runs.db, where the Forkwise
dashboard shows it next to the demo agent's runs.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # forkwise package
sys.path.insert(0, str(HERE.parent.parent / "evidencerag"))  # evidencerag repo

from forkwise import Tracer  # noqa: E402

tracer = Tracer(str(HERE / "demo_runs.db"))

from app import main as evr_app  # noqa: E402
from src import answer as answer_module  # noqa: E402
from src import rerank as rerank_module  # noqa: E402


@tracer.step(kind="retriever")
def retrieve(question):
    fused = evr_app._retriever.hybrid_search(question, top_k=20)
    return [{"doc_id": chunk["doc_id"], "page": chunk["page"],
             "score": round(score, 4)} for chunk, score in fused[:5]]


@tracer.step(kind="ml")
def rerank(question, fused_top5):
    fused = evr_app._retriever.hybrid_search(question, top_k=20)
    ranked = rerank_module.rerank(question, fused, evr_app._get_reranker(), 5)
    return [{"doc_id": chunk["doc_id"], "page": chunk["page"],
             "score": round(score, 4)} for chunk, score in ranked]


@tracer.step(kind="rule")
def answer(question, ranked_top5):
    fused = evr_app._retriever.hybrid_search(question, top_k=20)
    ranked = rerank_module.rerank(question, fused, evr_app._get_reranker(), 5)
    result = answer_module.extractive_answer(
        question, ranked, evr_app._retriever.embedder)
    if result["status"] != "answered":
        return {"status": result["status"]}
    top = result["spans"][0]
    return {"status": "answered", "quote": top["text"],
            "citation": top["citation"]}


@tracer.pipeline("evidencerag-qa")
def evidencerag_qa(question):
    top5 = retrieve(question)
    ranked = rerank(question, top5)
    result = answer(question, ranked)
    tracer.mark_success(result.get("status") == "answered")
    return result


if __name__ == "__main__":
    if evr_app._retriever is None:
        evr_app.startup()
    out = evidencerag_qa(
        "How many Wikipedia passages does Dense Passage Retrieval index?")
    print("answer status:", out.get("status"))
    if out.get("citation"):
        print("citation:", out["citation"]["doc_id"],
              "page", out["citation"]["page"])
    print("runs in db:", len(tracer.store.list_runs()))
