"""Fork-and-diff replay: the feature the incumbents do not have.

Given a recorded run of a @tracer.pipeline, re-execute it from step N:
steps before N are served from the recording (their recorded outputs are
fed forward, so tools and retrievers are NOT called again), step N runs
live with optional overrides (a different model, different parameters),
and steps after N run live. The result is a new run plus a step-by-step
diff: what changed, the cost delta, and the latency delta.
"""

import json


def replay_run(tracer, run_id, from_seq, overrides=None):
    """Re-execute run `run_id` from step `from_seq`. Returns
    (new_run_dict, diff_dict). The pipeline must be registered with this
    tracer (i.e., its module imported in this process)."""
    original = tracer.store.get_run(run_id)
    if original is None:
        raise ValueError(f"no such run: {run_id}")
    spans_by_seq = {s["seq"]: s for s in tracer.store.get_spans(run_id)}
    if from_seq < 1 or from_seq > len(spans_by_seq) + 1:
        raise ValueError(f"run {run_id} has steps 1..{len(spans_by_seq)}; "
                         f"from_seq must be in 1..{len(spans_by_seq) + 1} "
                         "(last+1 replays fully from the recording)")
    from .sdk import GLOBAL_PIPELINES
    spans_list = tracer.store.get_spans(run_id)
    if from_seq == len(spans_list) + 1:
        # Full reproduction needs no live code: rebuild the run purely
        # from the recording. This is what makes imported traces (and
        # the pinned-pytest export) work anywhere.
        return _reproduce_from_recording(tracer, original, spans_list), \
            diff_runs(tracer, run_id, _last_run_id(tracer))
    raw_fn = tracer.registry.get("pipeline-fn:" + original["name"]) or \
        GLOBAL_PIPELINES.get(original["name"])
    if raw_fn is None:
        raise ValueError(
            f"This run has no live code loaded in this process (it was "
            f"imported from logs, or its module is not imported), so it "
            f"cannot be re-executed live. Two things still work: exact "
            f"reproduction from the recording (set From step to "
            f"{len(spans_list) + 1}) and the pytest export.")
    run_input = (json.loads(original["input_json"]) if original["input_json"]
                 else {"args": [], "kwargs": {}})
    replay = {"run_id": run_id, "from_seq": from_seq,
              "spans_by_seq": spans_by_seq, "overrides": overrides or {}}
    before = {r["id"] for r in tracer.store.list_runs(limit=100000)}
    tracer._execute_run(raw_fn, original["name"],
                        tuple(run_input.get("args", [])),
                        dict(run_input.get("kwargs", {})), replay=replay)
    new_runs = [r for r in tracer.store.list_runs(limit=100000)
                if r["id"] not in before]
    if not new_runs:
        raise RuntimeError("replay did not create a run")
    new_run = new_runs[0]
    return new_run, diff_runs(tracer, run_id, new_run["id"])


def diff_runs(tracer, original_id, replay_id):
    orig = {s["seq"]: s for s in tracer.store.get_spans(original_id)}
    new = {s["seq"]: s for s in tracer.store.get_spans(replay_id)}
    steps = []
    for seq in sorted(set(orig) | set(new)):
        o, n = orig.get(seq), new.get(seq)
        if not (o and n):
            continue
        steps.append({
            "seq": seq, "name": n["name"], "kind": n["kind"],
            "output_changed": (o["output_json"] or "") != (n["output_json"] or ""),
            "from_recording": '"from_recording": true' in (n["attrs_json"] or ""),
            "cost_delta": round((n["cost_usd"] or 0) - (o["cost_usd"] or 0), 6),
            "latency_delta_ms": round((n["latency_ms"] or 0) - (o["latency_ms"] or 0), 2),
            "original_output": o["output_json"],
            "replay_output": n["output_json"],
        })
    o_run, n_run = tracer.store.get_run(original_id), tracer.store.get_run(replay_id)
    return {
        "original_run": original_id, "replay_run": replay_id,
        "steps": steps,
        "total_cost_delta": round((n_run["total_cost_usd"] or 0) - (o_run["total_cost_usd"] or 0), 6),
        "total_latency_delta_ms": round((n_run["latency_ms"] or 0) - (o_run["latency_ms"] or 0), 2),
        "output_changed": (o_run["output_json"] or "") != (n_run["output_json"] or ""),
    }


def _last_run_id(tracer):
    return tracer.store.list_runs(limit=1)[0]["id"]


def _reproduce_from_recording(tracer, original, spans_list):
    import time as _time
    store = tracer.store
    new_id = store.create_run(
        original["name"],
        input_obj=json.loads(original["input_json"]) if original["input_json"] else None,
        replay_of=original["id"],
        replay_from_seq=len(spans_list) + 1)
    for span in spans_list:
        store.add_span({
            "id": store.new_id(), "run_id": new_id,
            "parent_id": span["parent_id"], "seq": span["seq"],
            "name": span["name"], "kind": span["kind"],
            "started_at": span["started_at"], "ended_at": span["ended_at"],
            "latency_ms": span["latency_ms"],
            "input": json.loads(span["input_json"]) if span["input_json"] else None,
            "output": json.loads(span["output_json"]) if span["output_json"] else None,
            "error": span["error"], "model": span["model"],
            "tokens_in": span["tokens_in"], "tokens_out": span["tokens_out"],
            "cost_usd": span["cost_usd"], "priced": span["priced"],
            "attrs": {"from_recording": True}})
    output = json.loads(original["output_json"]) if original["output_json"] else None
    store.finish_run(new_id, original["status"], original["success"], output,
                     _time.time(), original["latency_ms"])
    return store.get_run(new_id)
