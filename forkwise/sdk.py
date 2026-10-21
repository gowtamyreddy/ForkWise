"""Forkwise SDK: trace any plain-Python agent with one decorator.

    tracer = Tracer("runs.db")

    @tracer.pipeline("qa-agent")
    def answer(question):
        docs = retrieve(question)          # a @tracer.step
        return compose(question, docs)     # a @tracer.step

    @tracer.step(kind="retriever")
    def retrieve(question): ...

    @tracer.step(kind="llm")
    def compose(question, docs):
        ...
        report_usage(model="gpt-4o-mini", tokens_in=812, tokens_out=140)
        return text

No framework required; steps are ordinary functions. Every call is
recorded (inputs, outputs, latency, tokens, cost) into one SQLite file.
Runs made of @tracer.step calls are replayable: see replay.py.
"""

import contextvars
import functools
import json
import time

from .pricing import span_cost
from .store import KINDS, Store

_ctx = contextvars.ContextVar("forkwise_ctx", default=None)

# Pipelines register globally as well as per-Tracer, so any Tracer over
# the same database can replay a run once the defining module is imported.
GLOBAL_PIPELINES = {}


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def report_usage(model=None, tokens_in=None, tokens_out=None, cost_usd=None):
    """Attach model/token usage to the span currently executing."""
    ctx = _ctx.get()
    if ctx and ctx.get("span_stack"):
        top = ctx["span_stack"][-1]
        if model is not None:
            top["model"] = model
        if tokens_in is not None:
            top["tokens_in"] = tokens_in
        if tokens_out is not None:
            top["tokens_out"] = tokens_out
        if cost_usd is not None:
            top["extra_cost"] = cost_usd


class Tracer:
    def __init__(self, db_path="forkwise.db"):
        self.store = Store(db_path)
        self.registry = {}  # step name -> function (for replay)

    # ---------- runs ----------
    def pipeline(self, name=None):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return self._execute_run(fn, name or fn.__name__, args, kwargs)
            self.registry["pipeline:" + (name or fn.__name__)] = wrapper
            self.registry["pipeline-fn:" + (name or fn.__name__)] = fn
            GLOBAL_PIPELINES[name or fn.__name__] = fn
            return wrapper
        return deco

    def _execute_run(self, fn, run_name, args, kwargs, replay=None):
        run_input = {"args": [_jsonable(a) for a in args],
                     "kwargs": {k: _jsonable(v) for k, v in kwargs.items()}}
        run_id = self.store.create_run(
            run_name, input_obj=run_input,
            replay_of=replay["run_id"] if replay else None,
            replay_from_seq=replay["from_seq"] if replay else None,
        )
        ctx = {"tracer": self, "run_id": run_id, "seq": 0, "span_stack": [],
               "replay": replay, "success": None}
        token = _ctx.set(ctx)
        started = time.time()
        status, output, error = "ok", None, None
        try:
            output = fn(*args, **kwargs)
        except Exception as exc:  # recorded, then re-raised
            status, error = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ended = time.time()
            success = ctx["success"]
            if success is None:
                success = 1 if status == "ok" else 0
            self.store.finish_run(run_id, status, success, _jsonable(output),
                                  ended, (ended - started) * 1000)
            _ctx.reset(token)
        return output

    def mark_success(self, value=True):
        ctx = _ctx.get()
        if ctx:
            ctx["success"] = 1 if value else 0

    # ---------- steps ----------
    def step(self, kind="generic", name=None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

        def deco(fn):
            step_name = name or fn.__name__
            self.registry[step_name] = fn

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                ctx = _ctx.get()
                if ctx is None:
                    return fn(*args, **kwargs)  # untraced outside a run
                store = ctx["tracer"].store  # record into the run owner's store
                ctx["seq"] += 1
                seq = ctx["seq"]
                replay = ctx.get("replay")
                recorded = None
                if replay:
                    recorded = replay["spans_by_seq"].get(seq)
                if (replay and recorded and seq < replay["from_seq"]
                        and recorded["name"] == step_name):
                    out = json.loads(recorded["output_json"]) if recorded["output_json"] else None
                    store.add_span({
                        "id": store.new_id(), "run_id": ctx["run_id"],
                        "parent_id": ctx["span_stack"][-1]["id"] if ctx["span_stack"] else None,
                        "seq": seq, "name": step_name, "kind": recorded["kind"],
                        "started_at": time.time(), "ended_at": time.time(),
                        "latency_ms": recorded["latency_ms"],
                        "input": json.loads(recorded["input_json"]) if recorded["input_json"] else None, "output": out,
                        "model": recorded["model"], "tokens_in": recorded["tokens_in"],
                        "tokens_out": recorded["tokens_out"],
                        "cost_usd": recorded["cost_usd"], "priced": recorded["priced"],
                        "attrs": {"from_recording": True},
                    })
                    return out
                call_kwargs = dict(kwargs)
                if replay and seq == replay["from_seq"] and replay.get("overrides"):
                    call_kwargs.update(replay["overrides"])
                span = {"id": store.new_id(), "run_id": ctx["run_id"],
                        "parent_id": ctx["span_stack"][-1]["id"] if ctx["span_stack"] else None,
                        "seq": seq, "name": step_name, "kind": kind,
                        "started_at": time.time(),
                        "input": {"args": [_jsonable(a) for a in args],
                                  "kwargs": {k: _jsonable(v) for k, v in call_kwargs.items()}}}
                ctx["span_stack"].append(span)
                try:
                    result = fn(*args, **call_kwargs)
                    span["output"] = _jsonable(result)
                    return result
                except Exception as exc:
                    span["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    span["ended_at"] = time.time()
                    span["latency_ms"] = (span["ended_at"] - span["started_at"]) * 1000
                    cost, priced = span_cost(span.get("model"), span.get("tokens_in"),
                                              span.get("tokens_out"), span.get("extra_cost", 0.0))
                    span["cost_usd"], span["priced"] = cost, priced
                    ctx["span_stack"].pop()
                    store.add_span(span)
            return wrapper
        return deco
