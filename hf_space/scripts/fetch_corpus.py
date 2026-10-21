"""Download a small corpus of arXiv PDFs about RAG and LLM evaluation.

Uses the public arXiv API (export.arxiv.org). Papers that fail to download
are skipped and logged, so one bad ID never breaks the whole fetch.
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# Well-known papers on retrieval-augmented generation and LLM evaluation.
# IDs verified against the papers' own first pages on 2026-10-20. An earlier
# version of this list contained five wrong IDs (2002.08904, 2002.08525,
# 2104.08691, 2312.10997, and 2401.18059 labeled as CRAG); those files held
# unrelated papers and silently poisoned the corpus. See
# data/papers/CONTENT_NOTES.md for the full audit trail, and run
# scripts/validate_corpus.py after fetching to prove identity.
ARXIV_IDS = [
    "2005.11401",  # Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    "2002.08909",  # REALM: Retrieval-Augmented Language Model Pre-Training
    "2007.01282",  # Leveraging Passage Retrieval with Generative Models (FiD)
    "2004.04906",  # Dense Passage Retrieval for Open-Domain Question Answering
    "2112.04426",  # Improving Language Models by Retrieving from Trillions of Tokens (RETRO)
    "2309.15217",  # RAGAs: Automated Evaluation of Retrieval Augmented Generation
    "2307.03172",  # Lost in the Middle: How Language Models Use Long Contexts
    "2310.11511",  # Self-RAG: Learning to Retrieve, Generate, and Critique
    "2401.15884",  # Corrective Retrieval Augmented Generation (CRAG)
    "2306.05685",  # Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
    "2303.16634",  # G-Eval: NLG Evaluation using GPT-4
    "2401.18059",  # RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
    "2407.04051",  # FunAudioLLM: Voice Understanding and Generation Foundation Models
    "2410.08035",  # IntrinsicVoice: Real-Time Voice Interaction Abilities
    "2501.18096",  # LLMs can see and hear without any training (MILS)
    "2505.19134",  # Incentivizing High-Quality Human Annotations with Golden Questions
]

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
REQUEST_DELAY_SECONDS = 3
TIMEOUT_SECONDS = 60

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"


def fetch_metadata(arxiv_ids):
    """Ask the arXiv API for titles keyed by id. Returns {id: title}."""
    params = {"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)}
    response = requests.get(API_URL, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.text)
    titles = {}
    for entry in root.findall("atom:entry", namespace):
        id_url = entry.find("atom:id", namespace).text.strip()
        arxiv_id = id_url.rsplit("/abs/", 1)[-1].split("v")[0]
        title = entry.find("atom:title", namespace).text.strip().replace("\n", " ")
        titles[arxiv_id] = title
    return titles


def download_pdf(arxiv_id, destination):
    """Download one PDF. Returns True on success, False on any failure."""
    try:
        with requests.get(PDF_URL.format(arxiv_id=arxiv_id), stream=True,
                           timeout=TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            with open(destination, "wb") as handle:
                for piece in response.iter_content(chunk_size=65536):
                    handle.write(piece)
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means skip this paper
        print(f"  skipped {arxiv_id}: {exc}")
        if destination.exists():
            destination.unlink()
        return False


def _record_title(manifest, arxiv_id, title):
    """Record a fetched title without clobbering richer manifest entries.

    manifest.json v2 stores an object per document (title, authors, source,
    sha256). This script only knows the title, so it updates that one field
    and leaves the rest of the entry alone. A missing or empty title never
    overwrites a known one.
    """
    if not title:
        return
    entry = manifest.get(arxiv_id)
    if isinstance(entry, dict):
        entry["title"] = title
    else:
        manifest[arxiv_id] = title


def main():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching paper metadata from arXiv...")
    try:
        titles = fetch_metadata(ARXIV_IDS)
    except Exception as exc:  # noqa: BLE001 - continue with ids alone if API fails
        print(f"Metadata fetch failed ({exc}); continuing with ids only.")
        titles = {}

    manifest_path = PAPERS_DIR / "manifest.json"
    # Merge into the existing manifest: it also records documents that this
    # script does not fetch (e.g. user-contributed papers), and overwriting
    # it would silently delete their entries.
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloaded = 0
    for arxiv_id in ARXIV_IDS:
        destination = PAPERS_DIR / f"{arxiv_id}.pdf"
        if destination.exists():
            print(f"  already have {arxiv_id}")
            downloaded += 1
            _record_title(manifest, arxiv_id, titles.get(arxiv_id))
            continue
        print(f"  downloading {arxiv_id}...")
        if download_pdf(arxiv_id, destination):
            downloaded += 1
            _record_title(manifest, arxiv_id, titles.get(arxiv_id))
        time.sleep(REQUEST_DELAY_SECONDS)

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nDone: {downloaded}/{len(ARXIV_IDS)} PDFs in {PAPERS_DIR}")


if __name__ == "__main__":
    main()
