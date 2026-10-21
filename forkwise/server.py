"""Forkwise dashboard: local web UI + JSON API over the trace store.

Run:  uvicorn forkwise.server:app --port 8000
Env:  FORKWISE_DB (default: demo/demo_runs.db next to this package),
      FORKWISE_RUN_COST_CEILING, FORKWISE_RUN_LATENCY_CEILING_MS
"""

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forkwise.export_test import generate_test
from forkwise.importer import import_jsonl
from forkwise.replay import diff_runs, replay_run
from forkwise.sdk import Tracer
from forkwise.stats import finops

DB_PATH = os.environ.get(
    "FORKWISE_DB",
    str(Path(__file__).resolve().parent.parent / "demo" / "demo_runs.db"))
tracer = Tracer(DB_PATH)
store = tracer.store

# Register the demo pipeline so its runs can be replayed from the UI.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
try:
    import agent as demo_agent  # noqa: F401

    tracer.registry.update(demo_agent.tracer.registry)
except Exception:
    pass

app = FastAPI(title="Forkwise")


def _ceilings():
    cost = os.environ.get("FORKWISE_RUN_COST_CEILING")
    lat = os.environ.get("FORKWISE_RUN_LATENCY_CEILING_MS")
    return (float(cost) if cost else None, float(lat) if lat else None)


@app.get("/api/runs")
def api_runs():
    return store.list_runs()


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    return {"run": run, "spans": store.get_spans(run_id)}


@app.get("/api/stats")
def api_stats():
    cost, lat = _ceilings()
    return finops(store, run_cost_ceiling=cost, run_latency_ceiling_ms=lat)


@app.post("/api/import")
async def api_import(request: Request):
    text = (await request.body()).decode("utf-8")
    return import_jsonl(store, text)


@app.post("/api/replay")
async def api_replay(request: Request):
    body = await request.json()
    try:
        new_run, diff = replay_run(
            tracer, body["run_id"], int(body["from_seq"]),
            body.get("overrides") or {})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"new_run": new_run, "diff": diff}


@app.get("/api/runs/{run_id}/export-test")
def api_export_test(run_id: str):
    try:
        return PlainTextResponse(generate_test(store, run_id),
                                 media_type="text/x-python")
    except ValueError as exc:
        raise HTTPException(404, str(exc))


STYLE = """
body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:62rem;padding:0 1rem;color:#1a1a1a}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;font-size:.92rem}
.bar{height:1rem;background:#2f6fed;border-radius:.25rem}
code{background:#f2f2f2;padding:.1rem .3rem;border-radius:.25rem}
.card{border:1px solid #ddd;border-radius:.5rem;padding:1rem;margin:1rem 0}
"""


@app.get("/", response_class=HTMLResponse)
def index():
    cost, lat = _ceilings()
    stats = finops(store, run_cost_ceiling=cost, run_latency_ceiling_ms=lat)
    t = stats["totals"]
    rows = "".join(
        "<tr><td><a href='/run/{id}'>{id}</a></td><td>{name}</td>"
        "<td>{ok}</td><td>${cost:.6f}</td><td>{lat:.0f} ms</td></tr>".format(
            id=r["id"], name=r["name"],
            ok="success" if r["success"] else "failed",
            cost=r["total_cost_usd"] or 0, lat=r["latency_ms"] or 0)
        for r in store.list_runs())
    return f"""<!doctype html><title>Forkwise</title><style>{STYLE}</style>
<h1>Forkwise</h1>
<p>Local-first agent observability: every run, step, token, and cent, in one file.</p>
<div class="card"><b>{t['runs']}</b> runs &middot; <b>{t['succeeded']}</b> succeeded &middot;
total cost <b>${t['total_cost_usd']:.6f}</b> &middot; cost per successful task
<b>{('$%.6f' % t['cost_per_successful_task']) if t['cost_per_successful_task'] is not None else 'n/a'}</b> &middot;
failed-run waste <b>${t['failed_run_waste_usd']:.6f}</b> &middot; retry waste
<b>${t['retry_waste_usd']:.6f}</b></div>
<table><tr><th>run</th><th>pipeline</th><th>outcome</th><th>cost</th><th>latency</th></tr>{rows}</table>
"""



@app.get("/compare/{a}/{b}", response_class=HTMLResponse)
def compare(a: str, b: str):
    if store.get_run(a) is None or store.get_run(b) is None:
        raise HTTPException(404, "no such run")
    diff = diff_runs(tracer, a, b)
    rows = ""
    for step in diff["steps"]:
        rows += (
            "<tr><td>{seq}</td><td>{name}</td><td>{kind}</td>"
            "<td>{rec}</td><td>{chg}</td><td>{cd:+.6f}</td><td>{ld:+.1f} ms</td></tr>"
            "<tr class='detail'><td colspan='7'>"
            "<b>original output:</b> <code>{oo}</code><br>"
            "<b>replay output:</b> <code>{ro}</code></td></tr>"
        ).format(
            seq=step["seq"], name=step["name"], kind=step["kind"],
            rec="from recording" if step["from_recording"] else "ran live",
            chg="<b>changed</b>" if step["output_changed"] else "same",
            cd=step["cost_delta"], ld=step["latency_delta_ms"],
            oo=(step["original_output"] or "")[:400],
            ro=(step["replay_output"] or "")[:400])
    return f"""<!doctype html><title>Compare runs</title><style>{STYLE}</style>
<p><a href='/'>&larr; all runs</a></p>
<h1>Run diff</h1>
<p><a href='/run/{a}'>original {a}</a> &rarr; <a href='/run/{b}'>replay {b}</a></p>
<p>final output <b>{"CHANGED" if diff["output_changed"] else "unchanged"}</b> &middot;
cost delta <b>${diff["total_cost_delta"]:+.6f}</b> &middot;
latency delta <b>{diff["total_latency_delta_ms"]:+.1f} ms</b></p>
<table><tr><th>#</th><th>step</th><th>kind</th><th>source</th><th>output</th>
<th>cost &Delta;</th><th>latency &Delta;</th></tr>{rows}</table>
"""

@app.get("/run/{run_id}", response_class=HTMLResponse)
def run_detail(run_id: str):
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    spans = store.get_spans(run_id)
    imported = '"imported": true' in (run["meta_json"] or "")
    max_lat = max((s["latency_ms"] or 0 for s in spans), default=1) or 1
    def _pretty(raw):
        if not raw:
            return ""
        try:
            return json.dumps(json.loads(raw), indent=1)[:1500]
        except (ValueError, TypeError):
            return raw[:1500]

    rows = ""
    for s in spans:
        rows += (
            "<tr class='spanrow' onclick='toggleDetail({seq})'>"
            "<td>{seq}</td><td>{name}</td><td>{kind}</td>"
            "<td><div class='bar' style='width:{w:.0f}%'></div></td>"
            "<td>{lat:.1f} ms</td><td>${cost:.6f}</td><td>{tok}</td></tr>"
            "<tr class='detail' id='detail-{seq}' style='display:none'>"
            "<td colspan='7'><b>input</b><pre>{inp}</pre>"
            "<b>output</b><pre>{out}</pre>{err}{meta}</td></tr>"
        ).format(
            seq=s["seq"], name=s["name"], kind=s["kind"],
            w=100 * (s["latency_ms"] or 0) / max_lat,
            lat=s["latency_ms"] or 0, cost=s["cost_usd"] or 0,
            tok=(("%s in / %s out" % (s["tokens_in"], s["tokens_out"]))
                 if s["tokens_in"] is not None else ""),
            inp=_pretty(s["input_json"]), out=_pretty(s["output_json"]),
            err=("<b>error</b><pre>%s</pre>" % s["error"]) if s["error"] else "",
            meta=("<p>model: <code>%s</code> &middot; attrs: <code>%s</code></p>"
                  % (s["model"], s["attrs_json"])) if s["model"] or '"from_recording": true' in (s["attrs_json"] or "") else "")
    rows += """<script>
function toggleDetail(seq){
  var el = document.getElementById('detail-' + seq);
  el.style.display = (el.style.display === 'none') ? '' : 'none';
}
</script>"""
    step_count_plus = len(spans) + 1
    replay_links = ""
    if run["replay_of"]:
        replay_links = ("<p>This run is a replay of <a href='/run/%s'>%s</a> "
                        "&middot; <a href='/compare/%s/%s'><b>view the diff</b></a></p>"
                        % (run["replay_of"], run["replay_of"], run["replay_of"], run_id))
    children = [r for r in store.list_runs(limit=1000) if r["replay_of"] == run_id]
    if children:
        replay_links += "<p>Replays of this run: " + " &middot; ".join(
            "<a href='/compare/%s/%s'>%s</a>" % (run_id, ch["id"], ch["id"])
            for ch in children) + "</p>"
    imported_note = ("<p><b>Imported run:</b> this trace came from a log file, "
                     "so there is no live code to re-execute. Exact reproduction "
                     "from the recording and the pytest export still work.</p>"
                     if imported else "")
    return f"""<!doctype html><title>Run {run_id}</title><style>{STYLE}</style>
<p><a href='/'>&larr; all runs</a></p>
<h1>{run['name']}</h1>
<p>run <code>{run_id}</code> &middot; {"success" if run['success'] else "failed"} &middot;
${run['total_cost_usd']:.6f} &middot; {run['latency_ms'] or 0:.0f} ms
&middot; <a href='/api/runs/{run_id}/export-test'>download as pytest regression test</a></p>
<p>input: <code>{(run['input_json'] or '')[:300]}</code></p>
<p>output: <code>{(run['output_json'] or '')[:300]}</code></p>
<p>Click a step to inspect its inputs, outputs, and errors.</p>
<table><tr><th>#</th><th>step</th><th>kind</th><th></th><th>latency</th><th>cost</th><th>tokens</th></tr>{rows}</table>
{replay_links}
<div class="card"><h3>Fork and replay</h3>
{imported_note}<p>Re-run from a step: earlier steps are served from this recording,
the chosen step runs live with your overrides (JSON), later steps run live.
Setting From step to {step_count_plus} reproduces the run exactly from the
recording, with no live code needed.</p>
<label>From step <input id='seq' type='number' min='1' value='2'></label>
<label>Overrides <input id='ovr' size='40' value='{{}}'></label>
<button onclick='doReplay()'>Replay</button>
<pre id='diff'></pre></div>
<script>
async function doReplay(){{
  const res = await fetch('/api/replay', {{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{run_id: '{run_id}',
      from_seq: parseInt(document.getElementById('seq').value),
      overrides: JSON.parse(document.getElementById('ovr').value)}})}});
  const data = await res.json();
  document.getElementById('diff').textContent = JSON.stringify(data, null, 2);
  if (data.new_run) {{
    document.getElementById('diff').innerHTML +=
      "\n\nFull diff page: /compare/" + data.diff.original_run + "/" + data.new_run.id;
  }}
}}
</script>
"""
