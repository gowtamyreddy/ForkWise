"""Live wiring: EvidenceRAG, traced by Forkwise, continuously.

Run from the Forkwise repo root:

    python demo/live_qa.py            # dashboard + live /ask on :8001

Every POST /ask {"question": "..."} executes the real EvidenceRAG
pipeline (hybrid retrieval -> cross-encoder rerank -> extractive
answer) as a Forkwise-traced run and lands in the same database the
dashboard reads, so the runs list, the FinOps totals, and fork-and-diff
replay all fill up with live traffic instead of seed data. The rerank
step takes a top_k argument, so a live run can be forked from step 2
with a different top_k from the dashboard's replay box.

Env:
    EVIDENCERAG_HOME  path to the evidencerag repo
                      (default: <forkwise>/../evidencerag, sibling layout)
    FORKWISE_DB       trace database (default: demo/demo_runs.db)
    PORT              listen port (default: 8001)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
os.environ.setdefault("FORKWISE_DB", str(HERE / "demo_runs.db"))
sys.path.insert(0, str(REPO))

from fastapi import HTTPException, Request  # noqa: E402

from forkwise.server import app, tracer  # noqa: E402

EVIDENCERAG_HOME = Path(os.environ.get(
    "EVIDENCERAG_HOME", str(REPO.parent / "evidencerag")))

_evr = None
_rerank_module = None
_answer_module = None


def _ensure_loaded():
    """Import EvidenceRAG and build/load its index on first use."""
    global _evr, _rerank_module, _answer_module
    if _evr is not None:
        return
    if not EVIDENCERAG_HOME.exists():
        raise HTTPException(
            503, "EvidenceRAG repo not found at %s; set EVIDENCERAG_HOME"
            % EVIDENCERAG_HOME)
    sys.path.insert(0, str(EVIDENCERAG_HOME))
    from app import main as evr_app
    from src import answer as answer_module
    from src import rerank as rerank_module
    if evr_app._retriever is None:
        evr_app.startup()
    _evr, _rerank_module, _answer_module = evr_app, rerank_module, answer_module


@tracer.step(kind="retriever")
def retrieve(question):
    return _evr._retriever.hybrid_search(question, top_k=20)


@tracer.step(kind="ml")
def rerank(question, candidates, top_k=5):
    return _rerank_module.rerank(
        question, candidates, _evr._get_reranker(), top_k)


@tracer.step(kind="rule")
def answer(question, ranked):
    return _answer_module.extractive_answer(
        question, ranked, _evr._retriever.embedder)


@tracer.pipeline("evidencerag-live")
def evidencerag_live(question):
    _ensure_loaded()
    fused = retrieve(question)
    ranked = rerank(question, fused)
    result = answer(question, ranked)
    tracer.mark_success(result.get("status") == "answered")
    return result


@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    result = evidencerag_live(question)
    run = tracer.store.list_runs(limit=1)[0]
    return {"question": question, "answer": result,
            "run_id": run["id"], "run_page": "/run/%s" % run["id"]}


if __name__ == "__main__":
    import uvicorn

    _ensure_loaded()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
