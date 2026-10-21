"""Forkwise x EvidenceRAG — the LIVE demo.

A real question-answering system (EvidenceRAG: hybrid retrieval ->
cross-encoder rerank -> extractive answer with citations, over 120+
real ML papers) with Forkwise tracing every step as it happens. There
is no seed data in this demo: the first question a visitor asks is
run #1, and the FinOps counters fill up with live traffic. Any run can
be forked at the rerank step with a different top_k to see exactly
what changes.

Boot: if the paper index is missing, fetch the corpus and build it
(blocking, before the UI opens — the same recipe as the EvidenceRAG
Space). Free-tier storage is ephemeral, so the trace log resets when
the Space sleeps; that is stated in the UI rather than hidden.
"""
import subprocess
import sys
from pathlib import Path

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
    from src import rerank as rerank_module
    if evr_app._retriever is None:
        evr_app.startup()
    _evr, _rerank_module, _answer_module = (
        evr_app, rerank_module, answer_module)
    print("boot: EvidenceRAG ready; Forkwise tracing is live.", flush=True)


_boot()


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
    fused = retrieve(question)
    ranked = rerank(question, fused)
    result = answer(question, ranked)
    tracer.mark_success(result.get("status") == "answered")
    return result


def _fmt_answer(result):
    if result.get("status") != "answered":
        return ("**Insufficient evidence.** The best-matching passage "
                "scored below the answer threshold, so EvidenceRAG "
                "refused rather than guess — and Forkwise recorded the "
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
    "allowance — or run the whole thing locally from GitHub (the "
    "repo's live mode does exactly what this page does): "
    "https://github.com/gowtamyreddy/ForkWise")


def _is_quota_error(exc):
    text = str(exc).lower()
    return "quota" in text or "zerogpu" in text


@spaces.GPU(duration=60)
def ask(question):
    question = (question or "").strip()
    if not question:
        return ("Type a question about retrieval-augmented generation "
                "first.", [], "", "", _counters())
    try:
        result = evidencerag_live(question)
    except Exception as exc:
        if _is_quota_error(exc):
            return (QUOTA_MSG, [], "", "", _counters())
        raise
    run = tracer.store.list_runs(limit=1)[0]
    return (_fmt_answer(result), _steps_rows(run["id"]), run["id"],
            run["id"][:8], _counters())


@spaces.GPU(duration=60)
def fork(run_id, top_k):
    if not run_id:
        return ("Ask a question first — then you can fork its run.",
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
            lines.append("- Step %d %s — served from the recording "
                         "(not re-run)" % (s["seq"], s["name"]))
        else:
            lines.append("- Step %d %s — re-run live, output %s"
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


HEADER = """# Forkwise × EvidenceRAG — live

Ask a real question over 120+ real ML papers. There is no chatbot and
no canned demo data: **EvidenceRAG** (hybrid retrieval → cross-encoder
reranking → extractive answers with document/page/character citations)
answers you, and **Forkwise** traces every step as it happens — then
lets you rewind the run and re-run the reranking step with a
different `top_k` to see exactly what changes.

*Honest fine print: free hosting sleeps when idle, which resets the
trace log and rebuilds the paper index on wake (~8 min). Costs read
$0 because every model is open weights running locally — no paid
APIs, which is itself one of the things the trace makes visible.*

Code: [ForkWise](https://github.com/gowtamyreddy/ForkWise) ·
[EvidenceRAG](https://github.com/gowtamyreddy/evidencerag)
"""

with gr.Blocks(title="Forkwise x EvidenceRAG — Live") as demo:
    gr.Markdown(HEADER)
    counters = gr.Markdown(_counters())
    question = gr.Textbox(
        label="Ask the papers",
        placeholder="e.g. How many passages does the DPR knowledge "
                    "source contain?",
        lines=2)
    ask_btn = gr.Button("Ask — and trace it", variant="primary")
    answer_md = gr.Markdown(label="Answer")
    steps = gr.Dataframe(
        headers=["Step", "Name", "Kind", "Model", "Latency (ms)", "Cost"],
        datatype=["number", "str", "str", "str", "number", "str"],
        label="Traced steps of this run", interactive=False)
    with gr.Row():
        run_id_state = gr.State("")
        run_id_show = gr.Textbox(label="Run id (fork this run)",
                                 interactive=False)
        top_k = gr.Slider(1, 10, value=3, step=1,
                          label="Fork from rerank with top_k")
        fork_btn = gr.Button("Fork and replay")
    fork_md = gr.Markdown(label="Fork result")
    gr.Examples(
        examples=[
            ["How many passages does the DPR knowledge source contain?"],
            ["What are Self-RAG's reflection tokens?"],
            ["How does RAPTOR organize retrieved text?"],
            ["What is the airspeed velocity of an unladen swallow?"],
        ],
        inputs=question)
    ask_btn.click(ask, inputs=question,
                  outputs=[answer_md, steps, run_id_state, run_id_show,
                           counters])
    question.submit(ask, inputs=question,
                    outputs=[answer_md, steps, run_id_state, run_id_show,
                             counters])
    fork_btn.click(fork, inputs=[run_id_state, top_k],
                   outputs=[fork_md, steps, counters])

if __name__ == "__main__":
    demo.queue()
    demo.launch()
