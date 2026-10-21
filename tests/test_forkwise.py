"""Forkwise test suite. Run: python -m pytest tests/ -q"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forkwise import Tracer, report_usage
from forkwise.export_test import generate_test
from forkwise.importer import import_jsonl
from forkwise.replay import replay_run
from forkwise.stats import finops

CALLS = {"retrieve": 0, "compose": 0}


def make_tracer(tmp_path):
    tracer = Tracer(str(tmp_path / "t.db"))

    @tracer.step(kind="retriever")
    def retrieve(question, top_k=2):
        CALLS["retrieve"] += 1
        return {"question": question, "top_k": top_k,
                "docs": ["doc-a", "doc-b"][:top_k]}

    @tracer.step(kind="llm")
    def compose(question, docs, model="gpt-4o-mini"):
        CALLS["compose"] += 1
        report_usage(model=model, tokens_in=100, tokens_out=25)
        return "%s|%s|%s" % (model, question, ",".join(docs["docs"]))

    @tracer.pipeline("toy-qa")
    def toy_qa(question, top_k=2):
        docs = retrieve(question, top_k=top_k)
        return compose(question, docs)

    return tracer, toy_qa


def test_trace_records_spans_and_cost(tmp_path):
    tracer, toy_qa = make_tracer(tmp_path)
    out = toy_qa("what?")
    assert out == "gpt-4o-mini|what?|doc-a,doc-b"
    runs = tracer.store.list_runs()
    assert len(runs) == 1 and runs[0]["success"] == 1
    spans = tracer.store.get_spans(runs[0]["id"])
    assert [s["name"] for s in spans] == ["retrieve", "compose"]
    assert spans[0]["kind"] == "retriever" and spans[1]["kind"] == "llm"
    # gpt-4o-mini: 100 in @ $0.15/M + 25 out @ $0.60/M = 0.000015 + 0.000015
    assert spans[1]["cost_usd"] == pytest.approx(0.00003, abs=1e-9)
    assert runs[0]["total_cost_usd"] == pytest.approx(0.00003, abs=1e-9)


def test_fork_replay_serves_earlier_steps_from_recording(tmp_path):
    tracer, toy_qa = make_tracer(tmp_path)
    CALLS.update(retrieve=0, compose=0)
    toy_qa("what?")
    assert CALLS == {"retrieve": 1, "compose": 1}
    run_id = tracer.store.list_runs()[0]["id"]
    new_run, diff = replay_run(tracer, run_id, from_seq=2,
                               overrides={"model": "gpt-4o"})
    # Step 1 came from the recording: retrieve was NOT called again.
    assert CALLS["retrieve"] == 1
    assert CALLS["compose"] == 2
    assert diff["steps"][0]["from_recording"] is True
    assert diff["steps"][1]["output_changed"] is True
    assert diff["output_changed"] is True
    assert new_run["output_json"] and "gpt-4o|what?" in new_run["output_json"]
    # gpt-4o is pricier than gpt-4o-mini -> positive cost delta.
    assert diff["total_cost_delta"] > 0


def test_full_replay_from_recording_reproduces_output(tmp_path):
    tracer, toy_qa = make_tracer(tmp_path)
    toy_qa("what?")
    run_id = tracer.store.list_runs()[0]["id"]
    spans = tracer.store.get_spans(run_id)
    new_run, diff = replay_run(tracer, run_id, from_seq=len(spans) + 1)
    assert diff["output_changed"] is False
    assert new_run["output_json"] == tracer.store.get_run(run_id)["output_json"]


def test_failed_run_marked_and_waste_counted(tmp_path):
    tracer = Tracer(str(tmp_path / "t.db"))

    @tracer.step(kind="llm")
    def spendy():
        report_usage(model="gpt-4o", tokens_in=1000, tokens_out=0)
        raise RuntimeError("boom")

    @tracer.pipeline("failer")
    def failer():
        spendy()

    with pytest.raises(RuntimeError):
        failer()
    stats = finops(tracer.store)
    assert stats["totals"]["failed"] == 1
    assert stats["totals"]["failed_run_waste_usd"] == pytest.approx(0.0025, abs=1e-9)
    assert stats["totals"]["cost_per_successful_task"] is None


def test_import_and_retry_waste(tmp_path):
    tracer = Tracer(str(tmp_path / "t.db"))
    fixture = (Path(__file__).resolve().parent.parent
               / "demo" / "sample_traces.jsonl").read_text()
    result = import_jsonl(tracer.store, fixture)
    assert result == {"runs": 4, "spans": 15}
    stats = finops(tracer.store)
    totals = stats["totals"]
    assert totals["runs"] == 4 and totals["succeeded"] == 3
    assert totals["cost_per_successful_task"] is not None
    # The failed run repeats search_kb 3x with identical input: 2 retries.
    assert len([e for e in stats["retry_events"] if e["step"] == "search_kb"]) == 2


def test_export_test_generates_runnable_pytest(tmp_path):
    tracer, toy_qa = make_tracer(tmp_path)
    toy_qa("what?")
    run_id = tracer.store.list_runs()[0]["id"]
    code = generate_test(tracer.store, run_id)
    assert "def test_pinned_trace_reproduces" in code
    assert run_id in code
    compile(code, "generated_test.py", "exec")  # syntactically valid


def test_budget_flags(tmp_path):
    tracer, toy_qa = make_tracer(tmp_path)
    toy_qa("what?")
    stats = finops(tracer.store, run_cost_ceiling=0.00001)
    assert len(stats["over_budget"]) == 1
    assert "cost" in stats["over_budget"][0]["reasons"][0]


def test_imported_run_reproduces_from_recording_without_code(tmp_path):
    # No pipeline is registered on this tracer: live replay must refuse
    # with a clear message, but exact reproduction from the recording
    # (from_seq = last step + 1) must work.
    tracer = Tracer(str(tmp_path / "t.db"))
    fixture = (Path(__file__).resolve().parent.parent
               / "demo" / "sample_traces.jsonl").read_text()
    import_jsonl(tracer.store, fixture)
    run = next(r for r in tracer.store.list_runs()
               if r["name"] == "support-agent" and r["success"])
    spans = tracer.store.get_spans(run["id"])
    with pytest.raises(ValueError, match="no live code"):
        replay_run(tracer, run["id"], from_seq=2)
    new_run, diff = replay_run(tracer, run["id"], from_seq=len(spans) + 1)
    assert diff["output_changed"] is False
    assert diff["total_cost_delta"] == 0
    assert all(step["from_recording"] for step in diff["steps"])
    assert new_run["output_json"] == run["output_json"]
    assert new_run["replay_of"] == run["id"]
