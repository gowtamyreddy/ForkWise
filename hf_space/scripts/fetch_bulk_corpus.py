"""Bulk-fetch ~110 recent arXiv papers to grow the demo corpus past 100 docs.

Queries the public arXiv API for the most recent submissions across the
cs.CL / cs.LG / cs.AI / cs.CV categories, skips papers already in
data/papers/, and downloads PDFs at arXiv's requested pace (3 s between
requests). Writes data/papers/bulk_manifest.json with {id: title}.

This script exists so the demo can run at a more realistic corpus size
(100+ documents) while staying reproducible: anyone can run it and rebuild
the index with scripts/build_index.py. The graded golden set stays grounded
in the original curated papers (scripts/fetch_corpus.py); these bulk papers
act as realistic distractors.
"""

import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
REQUEST_DELAY_SECONDS = 3
TIMEOUT_SECONDS = 60
TARGET_NEW = 110
CANDIDATE_POOL = 220  # ask for more than needed; dedup and failures eat some

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"


S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


def fetch_candidates():
    """Candidate papers: prefer the baked list at data/bulk_ids.tsv
    (tab-separated arXiv ID + title, committed to the repo). The public
    search APIs (Semantic Scholar, arXiv) rate-limit shared and residential
    IPs unpredictably, so the script only falls back to a live lookup when
    the baked list is missing. Returns [(arxiv_id, title)]."""
    baked = PROJECT_ROOT / "data" / "bulk_ids.tsv"
    if baked.exists():
        candidates = []
        for line in baked.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                arxiv_id, title = line.split("\t", 1)
                candidates.append((arxiv_id.strip(), title.strip()))
        if candidates:
            print(f"  Using baked candidate list ({len(candidates)} IDs from data/bulk_ids.tsv).")
            return candidates
    print("  No baked list found; trying Semantic Scholar...")
    return _fetch_candidates_semantic_scholar()


def _fetch_candidates_semantic_scholar():
    """Live fallback: recent CS papers with arXiv IDs via Semantic Scholar."""
    candidates = []
    params = {
        "fields": "title,externalIds,year",
        "year": "2024-2026",
        "fieldsOfStudy": "Computer Science",
        "sort": "publicationDate:desc",
        "limit": 1000,
    }
    token = None
    try:
        for _ in range(3):  # 3 pages x 1000 is plenty
            if token:
                params["token"] = token
            response = requests.get(S2_URL, params=params, timeout=TIMEOUT_SECONDS)
            if response.status_code == 429:
                print("  S2 rate-limited; waiting 30s")
                time.sleep(30)
                response = requests.get(S2_URL, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            for paper in payload.get("data", []):
                arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
                if arxiv_id:
                    candidates.append((arxiv_id, paper.get("title", "")))
                    if len(candidates) >= CANDIDATE_POOL:
                        return candidates
            token = payload.get("token")
            if not token:
                break
    except requests.RequestException as exc:
        print(f"  Semantic Scholar lookup failed ({exc}); falling back to arXiv search API.")
        return _fetch_candidates_arxiv_api()
    return candidates


def _fetch_candidates_arxiv_api():
    """Fallback: most recent submissions across four CS categories, with
    exponential backoff (the arXiv API 429s shared egress IPs)."""
    query = "cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:cs.CV"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": CANDIDATE_POOL,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = None
    for attempt in range(6):
        response = requests.get(API_URL, params=params, timeout=TIMEOUT_SECONDS)
        if response.status_code == 200:
            break
        wait = min(60 * (2 ** attempt), 600)
        print(f"  arXiv API returned {response.status_code}; waiting {wait}s (attempt {attempt + 1}/6)")
        time.sleep(wait)
    if response.status_code != 200:
        raise requests.HTTPError(f"arXiv API still failing after retries ({response.status_code})")
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.text)
    candidates = []
    for entry in root.findall("atom:entry", namespace):
        id_url = entry.find("atom:id", namespace).text.strip()
        arxiv_id = id_url.rsplit("/abs/", 1)[-1].split("v")[0]
        title = " ".join(entry.find("atom:title", namespace).text.split())
        candidates.append((arxiv_id, title))
    return candidates


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


def main():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {path.stem for path in PAPERS_DIR.glob("*.pdf")}
    print(f"Corpus currently has {len(existing)} PDFs.")
    print("Fetching candidate list from arXiv...")
    candidates = fetch_candidates()
    fresh = [(aid, title) for aid, title in candidates if aid not in existing]
    random.Random(20261019).shuffle(fresh)  # spread across categories/dates
    print(f"{len(fresh)} new candidates; downloading up to {TARGET_NEW}...")

    manifest_path = PAPERS_DIR / "bulk_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    downloaded = 0
    for arxiv_id, title in fresh:
        if downloaded >= TARGET_NEW:
            break
        destination = PAPERS_DIR / f"{arxiv_id}.pdf"
        print(f"  downloading {arxiv_id}...")
        if download_pdf(arxiv_id, destination):
            downloaded += 1
            manifest[arxiv_id] = title
        time.sleep(REQUEST_DELAY_SECONDS)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = len(list(PAPERS_DIR.glob("*.pdf")))
    print(f"\nDone: {downloaded} new PDFs; corpus now has {total} PDFs in {PAPERS_DIR}")


if __name__ == "__main__":
    main()
