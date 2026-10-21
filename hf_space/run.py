"""Forkwise x EvidenceRAG: the LIVE demo.

A real question-answering system (EvidenceRAG: hybrid retrieval ->
cross-encoder rerank -> extractive answer with citations, over 120+
real ML papers) with Forkwise tracing every step as it happens. There
is no seed data in this demo: the first question a visitor asks is
run #1, and the FinOps counters fill up with live traffic. Any run can
be forked at the rerank step with a different top_k to see exactly
what changes.

Boot: if the paper index is missing, fetch the corpus and build it
(blocking, before the UI opens; the same recipe as the EvidenceRAG
Space). Free-tier storage is ephemeral, so the trace log resets when
the Space sleeps; that is stated in the UI rather than hidden.
"""
import os
import subprocess
import sys
from pathlib import Path

# ZeroGPU presents a virtual GPU, and sentence-transformers auto-selects
# CUDA when one is visible — which trips the ZeroGPU quota gate from inside
# a request. This demo is CPU-only by design: hide the GPU from torch
# before any library imports it.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import spaces  # noqa: E402

from forkwise import Tracer  # noqa: E402
from forkwise.replay import replay_run  # noqa: E402
from forkwise.stats import finops  # noqa: E402

DB_PATH = str(ROOT / "forkwise_live.db")
tracer = Tracer(DB_PATH)

_evr = None
_rerank_module = None
_answer_module = None
_reranker = None


def _boot():
    global _evr, _rerank_module, _answer_module
    artifacts = ROOT / "data" / "artifacts" / "index.faiss"
    if not artifacts.exists():
        for script in ("fetch_corpus.py", "fetch_bulk_corpus.py",
                       "build_index.py"):
            print("boot: running scripts/%s ..." % script, flush=True)
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                           cwd=str(ROOT), check=True)
    from app import main as evr_app
    from src import answer as answer_module
    from src import embed as embed_module
    from src import rerank as rerank_module
    # The embedder has the same disease as the reranker, silently:
    # built at startup, it auto-selects ZeroGPU's emulated CUDA
    # device and then produces wrong vectors at query time (junk
    # retrieval, every answer refused). Force its model class to
    # CPU before startup constructs it.
    import sentence_transformers as _st

    class _CPUSentenceTransformer(_st.SentenceTransformer):
        def __init__(self, *args, **kwargs):
            kwargs["device"] = "cpu"
            super().__init__(*args, **kwargs)

    embed_module.SentenceTransformer = _CPUSentenceTransformer
    if evr_app._retriever is None:
        evr_app.startup()
    _evr, _rerank_module, _answer_module = (
        evr_app, rerank_module, answer_module)
    # ZeroGPU presents an emulated CUDA device, and CrossEncoder
    # captures it as its target device at construction; predict()
    # then calls model.to(cuda) on every call, which reaches a raw
    # CUDA init outside any GPU allocation and dies. Pin the
    # reranker's target device to CPU.
    import torch
    global _reranker
    _reranker = rerank_module.Reranker()
    _reranker.model._target_device = torch.device("cpu")
    print("boot: EvidenceRAG ready; Forkwise tracing is live.", flush=True)


_boot()


@spaces.GPU(duration=10)
def _gpu_slot():
    """Defined so this Space stays schedulable on ZeroGPU hardware.

    It is never called from a visitor interaction: the demo's models
    are small enough to answer on CPU, so no click can ever spend or
    exhaust GPU quota.
    """
    return True


@tracer.step(kind="retriever")
def retrieve(question):
    return _evr._retriever.hybrid_search(question, top_k=20)


@tracer.step(kind="ml")
def rerank(question, candidates, top_k=5):
    return _rerank_module.rerank(
        question, candidates, _reranker, top_k)


@tracer.step(kind="rule")
def answer(question, ranked):
    return _answer_module.extractive_answer(
        question, ranked, _evr._retriever.embedder)


@tracer.pipeline("evidencerag-live")
def evidencerag_live(question):
    fused = retrieve(question)
    ranked = rerank(question, fused)
    result = answer(question, ranked)
    tracer.mark_success(result.get("status") == "answered")
    return result


def _fmt_answer(result):
    if result.get("status") != "answered":
        return ("**Insufficient evidence.** The best-matching passage "
                "scored below the answer threshold, so EvidenceRAG "
                "refused rather than guess. Forkwise recorded the "
                "run as a failure, which is what a real trace log should "
                "do with a refusal.")
    parts = []
    for span in result["spans"]:
        cite = span["citation"]
        parts.append("> %s" % span["text"])
        parts.append("*— %s, page %s, chars %s–%s, score %s*" % (
            cite["doc_id"], cite["page"], cite["char_start"],
            cite["char_end"], round(cite["score"], 3)))
    return "\n\n".join(parts)


def _steps_rows(run_id):
    rows = []
    for s in tracer.store.get_spans(run_id):
        rows.append([s["seq"], s["name"], s["kind"],
                     s["model"] or "local",
                     round(s["latency_ms"] or 0),
                     "$%.6f" % (s["cost_usd"] or 0)])
    return rows


def _counters():
    stats = finops(tracer.store)
    t = stats["totals"]
    cps = t["cost_per_successful_task"]
    cps_text = "$%.6f" % cps if cps is not None else "n/a yet"
    md = ("**Live trace log (since this demo woke up):** %d runs · "
          "%d answered · %d refused/failed · cost per successful task "
          "%s (every model here is open weights on local compute, "
          "so the honest total is $0 in API spend)"
          % (t["runs"], t["succeeded"], t["failed"], cps_text))
    return md


QUOTA_MSG = (
    "**The demo's free GPU quota is used up for now.** Hugging Face "
    "gives free Spaces a small daily GPU budget; it resets daily. "
    "Signing in with a free Hugging Face account gives you your own "
    "allowance. Or run the whole thing locally from GitHub (the "
    "repo's live mode does exactly what this page does): "
    "https://github.com/gowtamyreddy/ForkWise")


def _is_quota_error(exc):
    text = str(exc).lower()
    return "quota" in text or "zerogpu" in text


def ask(question):
    question = (question or "").strip()
    if not question:
        return ("Type a question about retrieval-augmented generation "
                "first.", [], "", "", _counters())
    try:
        result = evidencerag_live(question)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("ASK-FAILURE-DIAGNOSTIC: %r" % (exc,), flush=True)
        if _is_quota_error(exc):
            return (QUOTA_MSG, [], "", "", _counters())
        raise
    run = tracer.store.list_runs(limit=1)[0]
    return (_fmt_answer(result), _steps_rows(run["id"]), run["id"],
            run["id"][:8], _counters())


def fork(run_id, top_k):
    if not run_id:
        return ("Ask a question first, then you can fork its run.",
                [], _counters())
    try:
        new_run, diff = replay_run(tracer, run_id, 2, {"top_k": int(top_k)})
    except Exception as exc:
        if _is_quota_error(exc):
            return (QUOTA_MSG, [], _counters())
        raise
    lines = ["**Forked run %s from step 2 (rerank) with top_k=%d.**"
             % (new_run["id"][:8], int(top_k)), ""]
    for s in diff["steps"]:
        if s["from_recording"]:
            lines.append("- Step %d %s: served from the recording "
                         "(not re-run)" % (s["seq"], s["name"]))
        else:
            lines.append("- Step %d %s: re-run live, output %s"
                         % (s["seq"], s["name"],
                            "CHANGED" if s["output_changed"] else "same"))
    lines.append("")
    lines.append("**Final answer: %s.** Total latency delta %+.0f ms."
                 % ("CHANGED" if diff["output_changed"] else "unchanged",
                    diff["total_latency_delta_ms"]))
    spans = tracer.store.get_spans(new_run["id"])
    import json
    new_answer = json.loads(spans[-1]["output_json"])
    lines.append("")
    lines.append(_fmt_answer(new_answer))
    return ("\n".join(lines), _steps_rows(new_run["id"]), _counters())


HEADER = """# Forkwise \u00d7 EvidenceRAG: live demo

Ask a real question and watch a real system answer it. There is no
chatbot and no canned demo data. **EvidenceRAG** answers from 120+
ML papers, quoting the exact passages it used. **Forkwise** records
every step of the run as it happens.

**How to try it:** ask a question below. Look at the trace under
your answer. Then use the rewind box to re-run one step with a
different setting and see what changes."""

FINE_PRINT = """<small>Honest fine print: free hosting sleeps when
idle, which resets the trace log and rebuilds the paper index on
wake (about 8 min). Costs read $0 because every model is open
weights running locally, no paid APIs. It answers on CPU, a few
seconds per question, so there is no GPU quota to run out of.
Code: [ForkWise](https://github.com/gowtamyreddy/ForkWise) \u00b7
[EvidenceRAG](https://github.com/gowtamyreddy/evidencerag)</small>"""

ANSWER_PLACEHOLDER = ("_Your answer will appear here, quoted word "
                      "for word from the papers, with the page and "
                      "characters it came from._")

FORK_PLACEHOLDER = ("_After you ask a question, rewind its run here: "
                    "keep fewer passages in the ranking step and see "
                    "whether the answer changes._")

with gr.Blocks(title="Forkwise x EvidenceRAG: Live",
               theme=gr.themes.Soft(primary_hue="indigo",
                                    neutral_hue="slate"),
               css=".footer {display: none !important}") as demo:
    gr.Markdown(HEADER)
    counters = gr.Markdown(_counters())
    gr.Markdown("### 1. Ask a question")
    question = gr.Textbox(
        label="Your question about retrieval-augmented generation",
        placeholder="e.g. How many passages does the DPR knowledge "
                    "source contain?",
        lines=2)
    gr.Examples(
        examples=[
            ["How many passages does the DPR knowledge source contain?"],
            ["What are Self-RAG's reflection tokens?"],
            ["How does RAPTOR organize retrieved text?"],
            ["What is the airspeed velocity of an unladen swallow?"],
        ],
        inputs=question, label="Or tap one of these")
    ask_btn = gr.Button("Ask and trace it", variant="primary")
    gr.Markdown("### 2. The answer, and the trace behind it")
    answer_md = gr.Markdown(ANSWER_PLACEHOLDER, label="Answer")
    steps = gr.Dataframe(
        headers=["#", "Step", "Type", "Model", "Time (ms)", "Cost"],
        datatype=["number", "str", "str", "str", "number", "str"],
        label="Trace of this run: every step the system took",
        interactive=False)
    gr.Markdown("### 3. Rewind this run")
    gr.Markdown(
        "Re-run the ranking step keeping only the top few passages, "
        "and Forkwise replays the rest of the run from its recording. "
        "You see exactly which steps changed and whether the final "
        "answer did.")
    with gr.Row():
        top_k = gr.Slider(1, 10, value=3, step=1,
                          label="Keep this many passages (top_k)")
        fork_btn = gr.Button("Re-run from the ranking step")
    fork_md = gr.Markdown(FORK_PLACEHOLDER, label="Rewind result")
    with gr.Accordion("Run details", open=False):
        run_id_state = gr.State("")
        run_id_show = gr.Textbox(
            label="Run id of the last run (used by the rewind)",
            interactive=False)
    gr.Markdown(FINE_PRINT)
    ask_btn.click(ask, inputs=question,
                  outputs=[answer_md, steps, run_id_state, run_id_show,
                           counters], api_name="ask")
    question.submit(ask, inputs=question,
                    outputs=[answer_md, steps, run_id_state, run_id_show,
                             counters])
    fork_btn.click(fork, inputs=[run_id_state, top_k],
                   outputs=[fork_md, steps, counters], api_name="fork")

if __name__ == "__main__":
    demo.queue()
    demo.launch()
