"""FastAPI service: ask a question, get quoted spans with citations.

POST /query {"question": "...", "top_k": 5}
  -> {"status": "answered" | "insufficient_evidence",
      "answer_spans": [{"text": "...", "citation": {...}}],
      "citations": [...],
      "latency_ms": 123}

GET /health -> {"status": "ok", "chunks": N, "docs": [...], "corpus": "default" | "upload"}

GET / -> the demo UI (app/ui.html)

POST /upload (multipart form field "file", a .zip of PDFs)
  -> 202 {"status": "building", "docs": [...]}
  The zip replaces the active corpus: PDFs are ingested, embedded, and
  indexed in the background; GET /upload/status reports progress and
  /query answers 503 until the new index is swapped in.

POST /fetch {"url": "https://..."} -> 202 {"status": "building", "doc": "..."}
  Fetches a PDF or web page, appends it to the dynamic corpus, and rebuilds
  the index in the background. Repeats of the same URL get a numeric suffix.

POST /stt (multipart form field "file", recorded audio)
  -> {"text": "..."} transcribed locally by faster-whisper (base, int8 CPU).
  Powers the voice button: audio never leaves this machine.

POST /corpus/reset -> restore the built-in corpus from data/artifacts/.

The default corpus loads once at startup from data/artifacts/. Build it
first with: python scripts/build_index.py
"""

import io
import ipaddress
import re
import socket
import threading
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import answer as answer_module
from src import embed, index as vector_index, ingest, rerank, retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
UI_FILE = Path(__file__).resolve().parent / "ui.html"

MAX_ZIP_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_PDFS = 50
MAX_PDF_BYTES_EACH = 50 * 1024 * 1024  # one extracted PDF
MAX_EXTRACTED_TOTAL = 150 * 1024 * 1024  # whole zip, extracted (zip-bomb guard)
MAX_FETCH_BYTES = 25 * 1024 * 1024  # one /fetch download, enforced while streaming
MAX_FETCH_REDIRECTS = 5
FETCH_TIMEOUT = (5, 30)  # (connect, read) seconds

app = FastAPI(title="EvidenceRAG", version="0.2.0")

# The demo page is served from this app, but also allow file:// and local
# static servers during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_retriever = None
_default_retriever = None
_whisper_model = None
_reranker = None
_state_lock = threading.Lock()
_state = {"building": False, "corpus": "default", "error": None}


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The question to answer.")
    top_k: int = Field(5, ge=1, le=20, description="Retrieved chunks to consider.")
    rerank: bool = Field(True, description="Apply the cross-encoder rerank stage.")


class FetchRequest(BaseModel):
    url: str = Field(..., min_length=8, description="A PDF link or web page to add to the corpus.")


class Citation(BaseModel):
    doc_id: str
    page: int
    char_start: int
    char_end: int
    score: float


class AnswerSpan(BaseModel):
    text: str
    chunk_text: str
    rel_start: int
    rel_end: int
    citation: Citation


class QueryResponse(BaseModel):
    status: str
    answer_spans: list[AnswerSpan]
    citations: list[Citation]
    latency_ms: int
    reranked: bool


def build_retriever_from_dir(uploads_dir, embedder):
    """Run the full offline pipeline in memory over an uploads directory."""
    pdf_paths = sorted(uploads_dir.glob("*.pdf"))
    txt_paths = sorted(uploads_dir.glob("*.txt"))
    if not pdf_paths and not txt_paths:
        raise ValueError("No PDFs or text files found in the uploads directory.")
    all_chunks = []
    for pdf_path in pdf_paths:
        all_chunks.extend(ingest.chunk_pages(pdf_path.stem, ingest.read_pdf(pdf_path, pdf_path.stem)))
    for txt_path in txt_paths:
        all_chunks.extend(ingest.chunk_pages(txt_path.stem, ingest.read_text_file(txt_path)))
    if not all_chunks:
        raise ValueError("No extractable text found in the uploaded documents.")
    vectors = embedder.embed_passages([chunk["text"] for chunk in all_chunks])
    faiss_index = vector_index.build_index(vectors)
    return retrieve.Retriever(all_chunks, embedder, faiss_index)


def _unique_stem(uploads_dir, stem):
    """Avoid overwriting: 2309.15217 -> 2309.15217-2 if taken."""
    candidate, n = stem, 1
    while (uploads_dir / f"{candidate}.pdf").exists() or (uploads_dir / f"{candidate}.txt").exists():
        n += 1
        candidate = f"{stem}-{n}"
    return candidate


def _looks_like_pdf(data):
    """Content check, not extension trust: real PDFs start with %PDF-."""
    return data[:5] == b"%PDF-"


def _extract_pdfs(zip_bytes):
    """Extract valid PDF members of an uploaded zip to data/uploads/.

    Defenses, in order: member names are flattened to their basename, so a
    zip can never write outside the uploads directory (Zip Slip); each
    member's declared size and actual content are capped (zip bombs);
    members must actually be PDFs (magic bytes, not the .pdf extension).
    The previous dynamic corpus is only replaced once at least one valid
    PDF has been extracted, so a bad upload cannot wipe a working corpus.
    Returns the list of extracted paths.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise
    valid = []  # (filename, bytes), fully validated before anything is written
    total = 0
    seen_stems = set()
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name.lower().endswith(".pdf"):
                continue
            stem = Path(name).stem
            if stem in seen_stems:
                continue
            if info.file_size > MAX_PDF_BYTES_EACH:
                continue
            data = archive.read(info.filename)
            if len(data) > MAX_PDF_BYTES_EACH or not _looks_like_pdf(data):
                continue
            total += len(data)
            if total > MAX_EXTRACTED_TOTAL:
                break
            seen_stems.add(stem)
            valid.append((name, data))
            if len(valid) >= MAX_PDFS:
                break
    if not valid:
        return []
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(UPLOADS_DIR.glob("*.pdf")) + list(UPLOADS_DIR.glob("*.txt")):
        old.unlink()
    extracted = []
    for name, data in valid:
        target = UPLOADS_DIR / name
        target.write_bytes(data)
        extracted.append(target)
    return extracted


def _build_and_swap():
    """Background job: rebuild the retriever from data/uploads/ and swap it in."""
    global _retriever
    try:
        new_retriever = build_retriever_from_dir(UPLOADS_DIR, _default_retriever.embedder)
        with _state_lock:
            _retriever = new_retriever
            _state.update(building=False, corpus="upload", error=None)
    except Exception as exc:  # surfaced to the UI via /upload/status
        with _state_lock:
            _state.update(building=False, error=str(exc))


@app.on_event("startup")
def startup():
    global _retriever, _default_retriever
    chunks = ingest.load_chunks(ARTIFACTS_DIR / "chunks.jsonl")
    faiss_index = vector_index.load_index(ARTIFACTS_DIR / "index.faiss")
    embedder = embed.Embedder()
    _default_retriever = retrieve.Retriever(chunks, embedder, faiss_index)
    _retriever = _default_retriever


@app.get("/")
def home():
    return FileResponse(UI_FILE, media_type="text/html")


@app.get("/health")
def health():
    with _state_lock:
        state = dict(_state)
    docs = sorted({chunk["doc_id"] for chunk in _retriever.chunks})
    return {
        "status": "ok",
        "chunks": len(_retriever.chunks),
        "docs": docs,
        "corpus": state["corpus"],
        "building": state["building"],
    }


def _get_reranker():
    """Lazy-load the cross-encoder (downloads ~90 MB on first use)."""
    global _reranker
    if _reranker is None:
        _reranker = rerank.Reranker()
    return _reranker


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    with _state_lock:
        building = _state["building"]
    if building:
        raise HTTPException(status_code=503, detail="Index is building; try again shortly.")
    started = time.perf_counter()
    fused = _retriever.hybrid_search(request.question, top_k=20)
    reranked = False
    ranked = fused[: request.top_k]
    if request.rerank:
        ranked = rerank.rerank(request.question, fused, _get_reranker(), request.top_k)
        reranked = True
    result = answer_module.extractive_answer(request.question, ranked, _retriever.embedder)
    latency_ms = int((time.perf_counter() - started) * 1000)
    citations = [span["citation"] for span in result["spans"]]
    return QueryResponse(
        status=result["status"],
        answer_spans=result["spans"],
        citations=citations,
        latency_ms=latency_ms,
        reranked=reranked,
    )


@app.post("/upload", status_code=202)
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip file of PDFs.")
    zip_bytes = await file.read()
    if len(zip_bytes) > MAX_ZIP_BYTES:
        raise HTTPException(status_code=413, detail="Zip is larger than 100 MB.")
    try:
        pdf_paths = _extract_pdfs(zip_bytes)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="That file is not a valid zip.")
    if not pdf_paths:
        raise HTTPException(status_code=400, detail="No PDFs found in the zip.")
    with _state_lock:
        if _state["building"]:
            raise HTTPException(status_code=409, detail="An index build is already running.")
        _state.update(building=True, error=None)
    background_tasks.add_task(_build_and_swap)
    return {"status": "building", "docs": [path.stem for path in pdf_paths]}


class _TextExtractor(HTMLParser):
    """Minimal readable-text extractor: drop scripts, styles, noscript."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "header", "footer", "nav"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "header", "footer", "nav") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self):
        return ingest.clean_text(" ".join(self.parts))


def _doc_id_from_url(url):
    """Stable doc id for a fetched URL. arXiv ids keep their familiar form."""
    arxiv = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", url)
    if arxiv:
        return arxiv.group(1)
    parts = urlsplit(url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (parts.netloc + parts.path).strip("-/"))[:60].strip("-")
    return slug or "web-doc"


def _assert_public_host(hostname):
    """SSRF boundary: require EVERY address a host resolves to be public.

    Without this, /fetch can be pointed at loopback services or the cloud
    metadata endpoint (169.254.169.254) — directly, via a redirect from a
    public URL, or via an encoded IP form (decimal/octal), all of which
    getaddrinfo resolves to the real address. Fetched content lands in the
    corpus, where /query would read it back: a full exfiltration channel.
    Known residual risk: DNS rebinding between this check and the actual
    connect (requests re-resolves). Fully closing that window requires
    connecting to the pinned IP with manual SNI handling; the check still
    blocks the realistic attacks. Raises ValueError (user-safe message).
    """
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ValueError("That host could not be resolved.")
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise ValueError("That host could not be resolved.")
    for address in addresses:
        if not ipaddress.ip_address(address).is_global:
            raise ValueError(
                "Fetching private, loopback, link-local, or reserved addresses is not allowed."
            )


def _fetch_url_safely(url):
    """GET a URL with SSRF defenses. Returns (content_bytes, content_type).

    Redirects are followed manually (max MAX_FETCH_REDIRECTS) so every hop
    is re-validated; the body is streamed and aborted past MAX_FETCH_BYTES
    so a hostile server cannot exhaust memory by streaming forever.
    """
    current = url
    for _hop in range(MAX_FETCH_REDIRECTS + 1):
        parts = urlsplit(current)
        if parts.scheme not in ("http", "https"):
            raise ValueError("URL must start with http:// or https://.")
        if not parts.hostname:
            raise ValueError("URL has no host.")
        _assert_public_host(parts.hostname)
        response = requests.get(
            current,
            headers={"User-Agent": "EvidenceRAG demo (corpus fetch)"},
            timeout=FETCH_TIMEOUT,
            stream=True,
            allow_redirects=False,
        )
        try:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("That URL redirects without a destination.")
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            body = bytearray()
            for piece in response.iter_content(chunk_size=65536):
                body.extend(piece)
                if len(body) > MAX_FETCH_BYTES:
                    raise ValueError(
                        f"Fetched file is larger than {MAX_FETCH_BYTES // (1024 * 1024)} MB."
                    )
            return bytes(body), response.headers.get("Content-Type", "").lower()
        finally:
            response.close()
    raise ValueError("Too many redirects.")


@app.post("/fetch", status_code=202)
async def fetch_url(request: FetchRequest, background_tasks: BackgroundTasks):
    """Fetch a URL (PDF or web page) and add it to the dynamic corpus.

    The document is appended to data/uploads/ and the whole uploads folder is
    rebuilt in the background; poll GET /upload/status for progress. Repeats
    of the same URL get a numeric suffix instead of overwriting.
    """
    # State first: never download or write files while a build is running
    # (the builder reads this same directory; mutating it mid-build mixes
    # two corpus versions).
    with _state_lock:
        if _state["building"]:
            raise HTTPException(status_code=409, detail="An index build is already running.")
    try:
        content, content_type = _fetch_url_safely(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch that URL: {exc}")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if "application/pdf" in content_type or request.url.lower().endswith(".pdf"):
        if not _looks_like_pdf(content):
            raise HTTPException(
                status_code=400,
                detail="That URL claims to be a PDF but its content is not one.",
            )
        stem = _unique_stem(UPLOADS_DIR, _doc_id_from_url(request.url))
        target = UPLOADS_DIR / f"{stem}.pdf"
        target.write_bytes(content)
    elif "text/html" in content_type or "text/plain" in content_type or not content_type:
        extractor = _TextExtractor()
        extractor.feed(content.decode("utf-8", errors="replace"))
        text = extractor.text()
        if len(text) < 200:
            raise HTTPException(status_code=400, detail="That page has too little readable text to index.")
        stem = _unique_stem(UPLOADS_DIR, _doc_id_from_url(request.url))
        target = UPLOADS_DIR / f"{stem}.txt"
        target.write_text(text, encoding="utf-8")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type ({content_type}); send a PDF or an HTML page.",
        )

    with _state_lock:
        if _state["building"]:
            raise HTTPException(status_code=409, detail="An index build is already running.")
        _state.update(building=True, error=None)
    background_tasks.add_task(_build_and_swap)
    return {"status": "building", "doc": stem, "source": "pdf" if target.suffix == ".pdf" else "page"}


@app.get("/upload/status")
def upload_status():
    with _state_lock:
        state = dict(_state)
    docs = sorted({chunk["doc_id"] for chunk in _retriever.chunks})
    return {
        "building": state["building"],
        "corpus": state["corpus"],
        "error": state["error"],
        "docs": docs,
        "chunks": len(_retriever.chunks),
    }


def _decode_audio_to_array(path):
    """Decode any audio file to a 16 kHz mono float32 numpy array.

    faster-whisper's own decode path pins an old PyAV API (it passes
    metadata_errors= to av.open, removed in PyAV 15+). Decoding here with the
    stable PyAV API keeps the endpoint working across PyAV versions, and
    faster-whisper accepts a numpy array directly.
    """
    import av
    import numpy as np

    container = av.open(str(path))
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    pieces = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            pieces.append(resampled.to_ndarray().reshape(-1))
    if pieces:
        return np.concatenate(pieces).astype(np.float32)
    return np.zeros(0, dtype=np.float32)


def _get_whisper():
    """Lazy-load faster-whisper (base, int8 CPU) on first use.

    The model downloads once from HuggingFace (~150 MB) and is then cached,
    so the first transcription takes longer than the rest. Kept lazy because
    most sessions never touch the mic button.
    """
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Transcribe recorded audio (webm/wav/mp3) into text with faster-whisper.

    Called by the mic button in the demo UI. Audio never leaves this machine:
    unlike the browser's built-in speech recognition, nothing is sent to a
    vendor's cloud service.
    """
    import tempfile

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio recording.")
    if len(audio_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Recording is larger than 10 MB.")
    suffix = Path(file.filename or "recording").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        samples = _decode_audio_to_array(tmp.name)
        model = _get_whisper()
        segments, _info = model.transcribe(samples, language="en", vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No speech detected in the recording.")
    return {"text": text}


@app.post("/corpus/reset")
def corpus_reset():
    global _retriever
    with _state_lock:
        if _state["building"]:
            raise HTTPException(status_code=409, detail="An index build is already running.")
        _retriever = _default_retriever
        _state.update(corpus="default", error=None)
    return {"status": "ok", "corpus": "default"}
