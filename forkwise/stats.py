"""Agent FinOps: the rollups observability tools do not compute.

- cost per successful task (not just cost per run)
- wasted spend: cost of failed runs, and retry waste (the same step
  called again with identical input inside one run)
- latency percentiles per pipeline
- budget flags: runs over a cost or latency ceiling
"""

import hashlib

from .pricing import span_cost  # noqa: F401  (re-exported for callers)


def _pct(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round(p / 100 * (len(values) - 1)))))
    return round(values[k], 2)


def finops(store, run_cost_ceiling=None, run_latency_ceiling_ms=None):
    runs = store.all_runs()
    spans = store.all_spans()
    by_name = {}
    for run in runs:
        bucket = by_name.setdefault(run["name"], {
            "runs": 0, "succeeded": 0, "failed": 0, "total_cost_usd": 0.0,
            "wasted_cost_usd": 0.0, "latencies_ms": []})
        bucket["runs"] += 1
        bucket["total_cost_usd"] += run["total_cost_usd"] or 0
        if run["latency_ms"]:
            bucket["latencies_ms"].append(run["latency_ms"])
        if run["success"]:
            bucket["succeeded"] += 1
        else:
            bucket["failed"] += 1
            bucket["wasted_cost_usd"] += run["total_cost_usd"] or 0

    # Retry waste: within one run, a step invoked 2+ times with an
    # identical input hash; every invocation after the first is waste.
    spans_by_run = {}
    for span in spans:
        spans_by_run.setdefault(span["run_id"], []).append(span)
    retry_waste = 0.0
    retry_events = []
    for run_id, run_spans in spans_by_run.items():
        seen = {}
        for span in run_spans:
            digest = hashlib.sha256(
                ((span["name"] or "") + "|" + (span["input_json"] or "")).encode()
            ).hexdigest()[:12]
            if digest in seen:
                retry_waste += span["cost_usd"] or 0
                retry_events.append({"run_id": run_id, "step": span["name"],
                                     "seq": span["seq"], "cost_usd": span["cost_usd"]})
            else:
                seen[digest] = span["seq"]

    pipelines = {}
    for name, b in by_name.items():
        succeeded = b["succeeded"]
        pipelines[name] = {
            "runs": b["runs"], "succeeded": succeeded, "failed": b["failed"],
            "total_cost_usd": round(b["total_cost_usd"], 6),
            "cost_per_successful_task": (
                round(b["total_cost_usd"] / succeeded, 6) if succeeded else None),
            "failed_run_waste_usd": round(b["wasted_cost_usd"], 6),
            "latency_p50_ms": _pct(b["latencies_ms"], 50),
            "latency_p95_ms": _pct(b["latencies_ms"], 95),
        }

    over_budget = []
    for run in runs:
        reasons = []
        if run_cost_ceiling is not None and (run["total_cost_usd"] or 0) > run_cost_ceiling:
            reasons.append("cost ${:.4f} over ceiling ${:.4f}".format(
                run["total_cost_usd"], run_cost_ceiling))
        if run_latency_ceiling_ms is not None and (run["latency_ms"] or 0) > run_latency_ceiling_ms:
            reasons.append("latency {:.0f} ms over ceiling {:.0f} ms".format(
                run["latency_ms"], run_latency_ceiling_ms))
        if reasons:
            over_budget.append({"run_id": run["id"], "name": run["name"], "reasons": reasons})

    total_cost = round(sum(r["total_cost_usd"] or 0 for r in runs), 6)
    total_success = sum(1 for r in runs if r["success"])
    return {
        "totals": {
            "runs": len(runs), "succeeded": total_success,
            "failed": len(runs) - total_success, "total_cost_usd": total_cost,
            "cost_per_successful_task": (
                round(total_cost / total_success, 6) if total_success else None),
            "failed_run_waste_usd": round(
                sum(r["total_cost_usd"] or 0 for r in runs if not r["success"]), 6),
            "retry_waste_usd": round(retry_waste, 6),
        },
        "pipelines": pipelines,
        "retry_events": retry_events[:50],
        "over_budget": over_budget,
    }
