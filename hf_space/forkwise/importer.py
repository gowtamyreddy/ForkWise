"""Import traces recorded elsewhere, as JSONL.

One JSON object per line:
{"run": "support-agent", "success": true, "input": {...},
 "steps": [{"name": "classify", "kind": "llm", "model": "gpt-4o-mini",
            "tokens_in": 812, "tokens_out": 140, "latency_ms": 640,
            "input": {...}, "output": "..."}, ...]}

This is how a team with existing logs (or another tool's export) gets
FinOps rollups without re-instrumenting first.
"""

import json
import time

from .pricing import span_cost


def import_jsonl(store, text, source_name="import"):
    runs_created = 0
    spans_created = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        steps = record.get("steps", [])
        started = record.get("started_at") or (time.time() - sum(
            (s.get("latency_ms") or 0) for s in steps) / 1000)
        run_id = store.create_run(
            record.get("run") or source_name,
            input_obj=record.get("input"),
            tags=record.get("tags") or [],
            meta={"imported": True})
        total_latency = 0.0
        cursor = started
        for seq, step in enumerate(steps, start=1):
            latency = step.get("latency_ms") or 0
            total_latency += latency
            cost, priced = span_cost(step.get("model"), step.get("tokens_in"),
                                     step.get("tokens_out"), step.get("cost_usd") or 0.0)
            store.add_span({
                "id": store.new_id(), "run_id": run_id, "parent_id": None,
                "seq": seq, "name": step.get("name") or "step-%d" % seq,
                "kind": step.get("kind") or "generic",
                "started_at": cursor, "ended_at": cursor + latency / 1000,
                "latency_ms": latency, "input": step.get("input"),
                "output": step.get("output"), "error": step.get("error"),
                "model": step.get("model"), "tokens_in": step.get("tokens_in"),
                "tokens_out": step.get("tokens_out"), "cost_usd": cost,
                "priced": priced, "attrs": {"imported": True}})
            cursor += latency / 1000
            spans_created += 1
        success = record.get("success")
        store.finish_run(run_id, "ok" if success is not False else "error",
                         1 if success else 0, record.get("output"),
                         started + total_latency / 1000, total_latency)
        runs_created += 1
    return {"runs": runs_created, "spans": spans_created}
