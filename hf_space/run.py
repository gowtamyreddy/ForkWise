"""Forkwise live demo (Hugging Face Space, free tier).

A guided tour of the three things Forkwise does that incumbent tools
do not: fork-and-diff replay, cost per successful task, and tracing a
plain-Python agent with one decorator. Everything runs locally in this
Space over a small built-in knowledge base; the priced examples come
from the bundled sample-trace import, labeled as an import.

The full dashboard (uvicorn forkwise.server:app) ships in the GitHub
repo for local use; this panel is the visitor-friendly front door.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

try:
    import spaces

    gpu = spaces.GPU(duration=60)
except Exception:  # local runs without the spaces package
    def gpu(fn):
        return fn

import agent as demo_agent
from forkwise.importer import import_jsonl
from forkwise.replay import replay_run
from forkwise.stats import finops

tracer = demo_agent.tracer
store = tracer.store


def _seed():
    if store.list_runs(limit=1):
        return
    demo_agent.knowledge_qa("How many moons does Jupiter have?")
    demo_agent.knowledge_qa("Who was the first computer programmer?")
    demo_agent.knowledge_qa("What is the population of Mars?")
    import_jsonl(store, (ROOT / "demo" / "sample_traces.jsonl").read_text(),
                 source_name="support-agent-import")


def _span_rows(run_id):
    rows = []
    for s in store.get_spans(run_id):
        tok = ""
        if s["tokens_in"] is not None:
            tok = f"{s['tokens_in']} in / {s['tokens_out']} out"
        rows.append([s["seq"], s["name"], s["kind"],
                     round(s["latency_ms"] or 0, 1),
                     f"${s['cost_usd']:.6f}", tok])
    return rows


def _finops_md():
    stats = finops(store)
    t = stats["totals"]
    lines = [
        f"**{t['runs']} runs** - {t['succeeded']} succeeded, {t['failed']} failed - "
        f"total cost **${t['total_cost_usd']:.6f}** - "
        f"cost per successful task **{('$%.6f' % t['cost_per_successful_task']) if t['cost_per_successful_task'] is not None else 'n/a'}** - "
        f"failed-run waste **${t['failed_run_waste_usd']:.6f}** - "
        f"retry waste **${t['retry_waste_usd']:.6f}** ({len(stats['retry_events'])} retry events)",
        "",
    ]
    for name, p in stats["pipelines"].items():
        cps = ("$%.6f" % p["cost_per_successful_task"]) if p["cost_per_successful_task"] is not None else "n/a"
        lines.append(f"- **{name}**: {p['runs']} runs, {p['succeeded']} ok, "
                     f"cost/successful task {cps}, p95 latency {p['latency_p95_ms']:.0f} ms")
    return "\n".join(lines)


@gpu
def run_agent(question, style):
    answer = demo_agent.knowledge_qa(question, style=style)
    run = next(r for r in store.list_runs(limit=50)
               if r["name"] == "knowledge-qa")
    return answer, _span_rows(run["id"]), run["id"], _finops_md()


@gpu
def fork_run(run_id, from_seq, style):
    if not run_id:
        return "Run the agent first, then fork its run.", _finops_md()
    try:
        new_run, diff = replay_run(tracer, run_id, int(from_seq),
                                   overrides={"style": style})
    except ValueError as exc:
        return str(exc), _finops_md()
    lines = [f"**Final output {'CHANGED' if diff['output_changed'] else 'unchanged'}** - "
             f"cost delta ${diff['total_cost_delta']:+.6f} - "
             f"latency delta {diff['total_latency_delta_ms']:+.1f} ms", ""]
    for step in diff["steps"]:
        source = "from recording" if step["from_recording"] else "ran live"
        lines.append(f"- step {step['seq']} **{step['name']}** ({source}): "
                     f"output {'changed' if step['output_changed'] else 'same'}, "
                     f"cost {step['cost_delta']:+.6f}, latency {step['latency_delta_ms']:+.1f} ms")
    lines += ["", f"New run id: `{new_run['id']}` (a replay of `{run_id}`)"]
    return "\n".join(lines), _finops_md()


def build_app():
    import gradio as gr

    with gr.Blocks(title="Forkwise") as blocks:
        gr.Markdown(
            "# Forkwise - agent observability with a do-over button\n"
            "Every tool in this market lets you *watch* a failed agent run. "
            "Forkwise lets you **re-run it from the step that broke**, with a "
            "fix applied, and diff the result. Local-first: one decorator, one "
            "SQLite file, no account, no framework. Below is the real thing, "
            "running a plain-Python agent live.")
        with gr.Row():
            question = gr.Textbox(
                label="Ask the demo agent",
                value="How many moons does Jupiter have?")
            style = gr.Dropdown(["brief", "detailed"], value="brief",
                                label="Answer style")
        run_btn = gr.Button("Run agent", variant="primary")
        answer = gr.Markdown()
        spans = gr.Dataframe(
            headers=["#", "step", "kind", "latency ms", "cost", "tokens"],
            label="Traced steps of that run")
        run_id = gr.Textbox(label="Run id (fork this run below)")
        with gr.Row():
            from_seq = gr.Number(value=3, precision=0,
                                 label="Fork from step")
            fork_style = gr.Dropdown(["brief", "detailed"], value="detailed",
                                     label="Override: answer style")
        fork_btn = gr.Button("Fork and replay")
        diff = gr.Markdown()
        gr.Markdown("## FinOps across every run in this demo")
        fin = gr.Markdown()
        refresh = gr.Button("Refresh FinOps")
        run_btn.click(run_agent, [question, style],
                      [answer, spans, run_id, fin])
        fork_btn.click(fork_run, [run_id, from_seq, fork_style], [diff, fin])
        refresh.click(lambda: _finops_md(), None, fin)
        blocks.load(lambda: _finops_md(), None, fin)
    return blocks


if __name__ == "__main__":
    _seed()
    build_app().queue().launch(server_name="0.0.0.0", server_port=7860)
