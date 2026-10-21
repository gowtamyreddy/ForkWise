"""Seed the demo database: run the knowledge-QA agent three times and
import the priced sample support-agent traces."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent  # noqa: E402
from forkwise.importer import import_jsonl  # noqa: E402

if __name__ == "__main__":
    agent.knowledge_qa("How many moons does Jupiter have?")
    agent.knowledge_qa("Who was the first computer programmer?")
    agent.knowledge_qa("What is the population of Mars?")
    result = import_jsonl(
        agent.tracer.store,
        (Path(__file__).parent / "sample_traces.jsonl").read_text(),
        source_name="support-agent-import")
    runs = agent.tracer.store.list_runs()
    print("imported:", result)
    print("total runs in demo db:", len(runs))
    for run in runs:
        print(" ", run["id"], run["name"], "success=", run["success"],
              "cost=", round(run["total_cost_usd"], 6))
