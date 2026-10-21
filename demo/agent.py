"""Demo agent: a plain-Python research assistant over a small built-in
knowledge base. No framework, no LLM API: the compose step assembles an
extractive answer locally and reports its real token counts at $0
(model "local"). Fork step 3 with style="detailed" to see replay diff
the answer, or fork step 2 with a different top_k.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forkwise import Tracer, report_usage

tracer = Tracer(str(Path(__file__).resolve().parent / "demo_runs.db"))

KNOWLEDGE = [
    "Jupiter has 95 officially recognized moons as of 2024, including the four Galilean moons Io, Europa, Ganymede, and Callisto.",
    "Ganymede is the largest moon in the solar system and is even bigger than the planet Mercury.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen inside plant cells.",
    "The Amazon rainforest produces a significant share of the world's oxygen through photosynthesis and stores vast amounts of carbon.",
    "The first computer programmer was Ada Lovelace, who wrote an algorithm for Charles Babbage's Analytical Engine in the 1840s.",
    "The Analytical Engine, designed by Charles Babbage, is considered the first concept of a general-purpose computer.",
    "Honey never spoils: sealed jars of honey found in ancient Egyptian tombs were still edible after 3,000 years.",
    "Octopuses have three hearts and blue blood, and their arms can taste what they touch.",
]


STOPWORDS = {"what", "is", "the", "of", "a", "an", "does", "do", "how",
             "who", "was", "were", "are", "in", "on", "has", "have"}


def _tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOPWORDS]


@tracer.step(kind="rule")
def classify_intent(question):
    q = question.lower()
    if any(w in q for w in ("how many", "how much")):
        return "count"
    if any(w in q for w in ("who", "first", "invent")):
        return "person"
    return "fact"


@tracer.step(kind="retriever")
def retrieve_passages(question, top_k=2):
    q = set(_tokens(question))
    scored = []
    for idx, passage in enumerate(KNOWLEDGE):
        overlap = len(q & set(_tokens(passage)))
        if overlap:
            scored.append((overlap, idx, passage))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [{"index": idx, "score": score, "text": text}
            for score, idx, text in scored[:top_k]]


@tracer.step(kind="llm")
def compose_answer(question, passages, style="brief"):
    prompt_words = len(_tokens(question)) + sum(len(_tokens(p["text"])) for p in passages)
    if not passages:
        answer = "I could not find anything about that in the knowledge base."
    elif style == "detailed":
        answer = " ".join(p["text"] for p in passages)
    else:
        answer = passages[0]["text"]
    report_usage(model="local", tokens_in=prompt_words,
                 tokens_out=len(_tokens(answer)))
    return answer


@tracer.step(kind="rule")
def verify_answer(answer, passages):
    if not passages:
        return {"grounded": False, "reason": "no passages retrieved"}
    grounded = any(p["text"] in answer for p in passages)
    return {"grounded": grounded,
            "reason": "answer quotes a retrieved passage" if grounded else "answer drifted from sources"}


@tracer.pipeline("knowledge-qa")
def knowledge_qa(question, top_k=2, style="brief"):
    classify_intent(question)
    passages = retrieve_passages(question, top_k=top_k)
    answer = compose_answer(question, passages, style=style)
    verdict = verify_answer(answer, passages)
    tracer.mark_success(bool(verdict["grounded"]))
    return answer


if __name__ == "__main__":
    print(knowledge_qa("How many moons does Jupiter have?"))
    print(knowledge_qa("Who was the first computer programmer?"))
    print(knowledge_qa("What is the population of Mars?"))
