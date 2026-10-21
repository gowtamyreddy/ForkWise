"""Single-file SQLite store. One database file holds every run and span:
no server, no account, no retention limit. That is the product."""

import json
import sqlite3
import threading
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    success INTEGER,
    started_at REAL NOT NULL,
    ended_at REAL,
    latency_ms REAL,
    total_cost_usd REAL NOT NULL DEFAULT 0,
    input_json TEXT,
    output_json TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    meta_json TEXT NOT NULL DEFAULT '{}',
    replay_of TEXT,
    replay_from_seq INTEGER
);
CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    parent_id TEXT,
    seq INTEGER NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'generic',
    started_at REAL NOT NULL,
    ended_at REAL,
    latency_ms REAL,
    input_json TEXT,
    output_json TEXT,
    error TEXT,
    model TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL NOT NULL DEFAULT 0,
    priced INTEGER NOT NULL DEFAULT 0,
    attrs_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
"""

KINDS = ("llm", "retriever", "tool", "ml", "rule", "generic")


def _dumps(obj):
    try:
        return json.dumps(obj, default=str)[:200_000]
    except (TypeError, ValueError):
        return json.dumps(str(obj)[:200_000])


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def new_id(self):
        return uuid.uuid4().hex[:16]

    def create_run(self, name, input_obj=None, tags=None, meta=None,
                   replay_of=None, replay_from_seq=None):
        run_id = self.new_id()
        import time
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (id, name, started_at, input_json, tags_json,"
                " meta_json, replay_of, replay_from_seq) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, name, time.time(), _dumps(input_obj) if input_obj is not None else None,
                 json.dumps(tags or []), json.dumps(meta or {}), replay_of, replay_from_seq),
            )
            self._conn.commit()
        return run_id

    def finish_run(self, run_id, status, success, output_obj, ended_at, latency_ms):
        with self._lock:
            total = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) AS c FROM spans WHERE run_id=?",
                (run_id,),
            ).fetchone()["c"]
            self._conn.execute(
                "UPDATE runs SET status=?, success=?, ended_at=?, latency_ms=?,"
                " total_cost_usd=?, output_json=? WHERE id=?",
                (status, success, ended_at, latency_ms, total,
                 _dumps(output_obj) if output_obj is not None else None, run_id),
            )
            self._conn.commit()
        return total

    def add_span(self, span):
        with self._lock:
            self._conn.execute(
                "INSERT INTO spans (id, run_id, parent_id, seq, name, kind, started_at,"
                " ended_at, latency_ms, input_json, output_json, error, model, tokens_in,"
                " tokens_out, cost_usd, priced, attrs_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (span["id"], span["run_id"], span.get("parent_id"), span["seq"],
                 span["name"], span.get("kind", "generic"), span["started_at"],
                 span.get("ended_at"), span.get("latency_ms"),
                 _dumps(span["input"]) if span.get("input") is not None else None,
                 _dumps(span["output"]) if span.get("output") is not None else None,
                 span.get("error"), span.get("model"), span.get("tokens_in"),
                 span.get("tokens_out"), span.get("cost_usd", 0.0),
                 1 if span.get("priced") else 0, json.dumps(span.get("attrs") or {})),
            )
            self._conn.commit()

    def get_run(self, run_id):
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit=100):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_spans(self, run_id):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM spans WHERE run_id=? ORDER BY seq", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def all_runs(self):
        with self._lock:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
        return [dict(r) for r in rows]

    def all_spans(self):
        with self._lock:
            rows = self._conn.execute("SELECT * FROM spans ORDER BY run_id, seq").fetchall()
        return [dict(r) for r in rows]
