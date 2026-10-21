"""Turn PDFs and text files into overlapping, citable chunks.

Each chunk records where it came from: doc_id, page number, and character
offsets within the page text. Those offsets are what make citations exact:
a citation points at a span of the source document, not just a document.

Design note: offsets are relative to the page the chunk came from, because
pypdf gives us text page by page. Chunk boundaries are computed on the
cleaned text, so cleaning must happen before chunking (it does).
"""

import json
import re

from pypdf import PdfReader

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100


def clean_text(text):
    """Collapse all whitespace to single spaces and strip the result.

    Also replaces lone UTF-16 surrogates with spaces. Some PDFs encode
    math glyphs (e.g. mathematical bold letters, U+1D400 block) as
    surrogate pairs, and pypdf can return the halves unpaired. A lone
    surrogate is legal in a Python string but cannot be encoded to
    UTF-8, so it crashes the embedding tokenizer downstream with a
    misleading type error (found the hard way at 130 papers: arXiv
    2606.22352). The original glyph is unrecoverable at that point,
    so a space is the honest replacement.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"[\ud800-\udfff]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(path, doc_id):
    """Read a PDF. Returns a list of (page_number, cleaned_text) tuples."""
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        pages.append((page_number, clean_text(raw)))
    return pages


def read_text_file(path):
    """Read a plain text file as a single page of text."""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return [(1, clean_text(handle.read()))]


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Split text into overlapping chunks.

    Returns a list of (chunk_text, char_start, char_end) tuples. The overlap
    exists so a sentence that straddles a boundary still appears whole in at
    least one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end]
        if piece.strip():
            chunks.append((piece, start, end))
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_pages(doc_id, pages, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Chunk every page of one document. Returns a list of chunk dicts."""
    chunks = []
    for page_number, page_text in pages:
        for text, char_start, char_end in chunk_text(page_text, chunk_size, overlap):
            chunks.append(
                {
                    "doc_id": doc_id,
                    "page": page_number,
                    "char_start": char_start,
                    "char_end": char_end,
                    "text": text,
                }
            )
    return chunks


def save_chunks(chunks, path):
    """Write chunks as JSONL, one chunk per line."""
    with open(path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")


def load_chunks(path):
    """Read chunks back from a JSONL file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
