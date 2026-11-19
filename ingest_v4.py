# ---------------------------------------------------------------------------
# Serlachius RAG prototype: corpus ingestion (ingest_v4.py)
#
# Part of the MBA thesis "Designing for Trustworthiness: An Action Design
# Research Study on AI Knowledge Retrieval System Redesign in a Cultural
# Institution" (Lauri Lumme, Haaga-Helia University of Applied Sciences, 2026).
#
# This script built the vector store used in BIE 3 and in the linguistic
# experiment: 29 publications, 11,962 chunks (run in October 2026).
# Comments were added on 19 November 2026 for publication. No code was
# changed: the code is as it was used in the study.
#
# Not published: the corpus (the Gösta Serlachius Fine Arts Foundation's
# publications), the vector store built from it, and the study's logs, which
# contain participants' questions. The interface and prompts are in Finnish;
# English translations are given in comments.
# ---------------------------------------------------------------------------
#
# v3 refers to the collection used in BIE 2, built by an earlier script
# (ingest.py, not published) that extracted text with pypdf.

"""
Build the serlachius_v4 collection.

Differences from ingest.py (v3), each deliberate and each recorded:

1. Extraction uses `pdftotext` instead of pypdf. pypdf preserves the soft hyphen
   and line break at a typeset line end, so "lähti-\nvät" became "lähti - vät" in
   v3 and every word hyphenated across a line was stored broken. pdftotext rejoins
   them. This is an extraction fix, not a cleaning rule: no regex is involved.

2. Reference apparatus (endnotes, bibliography) is excluded. Those pages are dense
   with archive citations whose stems collide with ordinary query terms — they are
   what crowded out the answering passage in a retrieval probe during
   development (a question about when Gösta Serlachius died).

3. Metadata carries the publication's real title, not a filename, because staff
   read it in the interface.

4. The collection is rebuilt, not appended to.

Chunking (800/150), embedding model, k and the prompt are UNCHANGED from v3.
One variable at a time: the corpus moves, the pipeline does not.
"""

# System dependency — not installable via pip:
#   poppler (provides pdftotext), used by ingest_v4.py for text extraction
#   macOS:          brew install poppler
#   Debian/Ubuntu:  apt install poppler-utils
import shutil
if shutil.which("pdftotext") is None:
    raise SystemExit("pdftotext not found. Install poppler: brew install poppler")

import re
import subprocess
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Paths assume the project lives in ~/serlachius-rag, as on the author's
# machine. Change them to run the code elsewhere.
# MUSEUM is the folder of PDFs supplied by the Foundation (not published).
# The folder on disk is "AI AGENTTI Copy"; the lower-case "copy" below works
# only on a case-insensitive file system such as macOS's default.
HOME       = Path.home() / "serlachius-rag"
MUSEUM     = HOME / "AI AGENTTI copy" / "SERLACHIUS JULKAISUT"
OLD_CORPUS = HOME / "old_corpus"
DB_DIR     = HOME / "store"
COLLECTION = "serlachius_v4"

# Chunks of 800 characters with 150 characters of overlap, so that a fact at
# a chunk boundary is not split off. An engineering default, not tuned.
CHUNK_SIZE, OVERLAP = 800, 150          # unchanged from v3


# ---------------------------------------------------------------- what goes in

# English editions of books already present in Finnish. multilingual-e5 embeds
# both languages into one space, so a Finnish query can retrieve the English
# passage and the model may answer from it. Excluded for the staff evaluation;
# a bilingual corpus in a multilingual embedding space is future work.
EXCLUDE_FILES = {
    "Patrons.pdf",
    "gpt_2_helene_schjerfbeck_eng.pdf",
}

# Additional material gathered by the researcher during BIE 2 (two web-page
# prints and an article), outside the museum's curated set. Not used in v4:
# the list is empty, and the corresponding cleaning rules below are disabled.
EXTRA_SOURCES = []
#    (OLD_CORPUS / "gosta_ihmisena.pdf",              "Gösta Serlachius ihmisenä (verkkosivu)"),
#    (OLD_CORPUS / "kapoor_artikkeli.pdf",            "Anish Kapoor Mäntässä (artikkeli)"),
#    (OLD_CORPUS / "joutomaiden_ihmiskohtaloita.pdf", "Joutomaiden ihmiskohtaloita"),
#]

# File-name keys of the "Göstan pieni taidekirjasto" (Gösta's little art
# library) booklet series, mapped to artist names for the displayed title.
ARTISTS = {
    "segerstråle": "Segerstråle", "helene_schjerfbeck_fi": "Helene Schjerfbeck",
    "thesleff": "Thesleff", "autere": "Autere", "simberg": "Simberg",
    "gallen-kallela": "Gallen-Kallela", "ahtila": "Ahtila", "sakkinen": "Sakkinen",
    "edelfelt": "Edelfelt", "magnus_enckell": "Magnus Enckell",
    "emil_wikström": "Emil Wikström",
}

import unicodedata

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def title_for(pdf: Path) -> str:
    """Publication title as a curator would name it — this is what staff read."""
    folder, stem = nfc(pdf.parent.name), nfc(pdf.stem)
    m = re.match(r"gpt_\d+_(.+)$", stem)
    if m:
        key = m.group(1)
        return f"Göstan pieni taidekirjasto: {ARTISTS.get(key, key)}"
    return folder

# ------------------------------------------------------------------ extraction

# Markers typical of endnotes and bibliographies: archive abbreviations,
# page references ("s. 12" = p. 12), Kirjeenvaihto (correspondence),
# arkisto (archive). A page in the last quarter of a document with six or
# more of them is treated as reference apparatus and skipped.
APPARATUS_RE = re.compile(r"\b(?:SM|GS:|s\.\s*\d|Kirjeenvaihto|arkisto)\b")

def apparatus_pages(pages, min_hits=6, tail_fraction=0.25):
    """Endnote/bibliography pages, detected only in the document's final quarter.
    Restricting to a trailing block stops a citation-heavy mid-book page being lost."""
    first_tail = int(len(pages) * (1 - tail_fraction))
    return {i for i, txt in enumerate(pages, 1)
            if i > first_tail and len(APPARATUS_RE.findall(txt)) >= min_hits}


# Disabled cleaning rules for the web-page prints (URLs, time stamps, site
# navigation). Not used in v4, which contains no web-page prints.
#WEB_FURNITURE = [
#    (re.compile(r"https?://\S+\s*\d+/\d+"), ""),
#    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?\b"), ""),
#    (re.compile(r"\bVerkkokauppa\b|\bETUSIVU\b|\bmenu\b"), ""),
#]

def extract_pages(pdf: Path):
    """[(page_number, text)] via pdftotext — one subprocess per document.
    pdftotext separates pages with a form feed and rejoins hyphenated words."""
    out = subprocess.run(["pdftotext", str(pdf), "-"],
                         capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"pdftotext failed on {pdf.name}: {out.stderr[:200]}")
    raw = out.stdout.split("\f")
    skip = apparatus_pages(raw)
    pages = []
    for i, text in enumerate(raw, 1):
        if i in skip:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if re.search(r"[^\W\d_]{3,}", text):
            pages.append((i, text))
    return pages, len(skip)

# Chunks never cross a page boundary, so every chunk carries a single page
# number for citation. A cut is moved back to the last space if that space
# falls after 70 % of the chunk, so that words are not split.
# Note: the page number is the PDF page, which may differ from the page
# number in the printed edition (thesis, section 4.3.4).
def chunk_page_text(text, page_num, doc_title):
    """Unchanged from v3."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        piece = text[start:end]
        if end < len(text):
            cut = piece.rfind(" ")
            if cut > CHUNK_SIZE * 0.7:
                piece, end = piece[:cut], start + cut
        chunks.append({"text": piece.strip(), "doc": doc_title, "page": page_num})
        start = end - OVERLAP
    return chunks


# ----------------------------------------------------------------------- build

def sources():
    found = []
    for pdf in sorted(MUSEUM.rglob("*.pdf")):
        if pdf.name in EXCLUDE_FILES:
            print(f"  excluded: {pdf.name}")
            continue
        found.append((pdf, title_for(pdf)))
    for path, title in EXTRA_SOURCES:
        if path.exists():
            found.append((path, title))
        else:
            print(f"  missing (skipped): {path}")
    return found


def main():
    emb = SentenceTransformer("intfloat/multilingual-e5-base")
    client = chromadb.PersistentClient(str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)
        print(f"removed existing '{COLLECTION}'")
    except Exception:
        pass
    db = client.create_collection(COLLECTION)

    docs = sources()
    print(f"\n{len(docs)} documents\n" + "-" * 78)
    total_chunks = total_pages = total_skipped = 0

    for pdf, title in docs:
        pages, skipped = extract_pages(pdf)
        chunks = []
        for page_num, text in pages:
            chunks.extend(chunk_page_text(text, page_num, title))
        if not chunks:
            print(f"  WARNING: nothing extracted from {pdf.name}")
            continue

        ids   = [f"{pdf.stem}-p{c['page']}-{i}" for i, c in enumerate(chunks)]
        texts = [c["text"] for c in chunks]
        meta  = [{"doc": c["doc"], "page": c["page"]} for c in chunks]
        # E5 "passage: " prefix; ask.py uses "query: " for questions.
        vecs  = emb.encode([f"passage: {t}" for t in texts],
                           batch_size=32, show_progress_bar=False).tolist()
        db.add(ids=ids, documents=texts, metadatas=meta, embeddings=vecs)

        total_chunks += len(chunks); total_pages += len(pages); total_skipped += skipped
        print(f"  {len(pages):>4} pp  {len(chunks):>5} chunks  "
              f"{'(-' + str(skipped) + ' apparatus)' if skipped else '':<18} {title}")

    print("-" * 78)
    print(f"{total_pages} pages ingested, {total_skipped} apparatus pages excluded")
    print(f"{total_chunks} chunks in '{COLLECTION}' (count: {db.count()})")


if __name__ == "__main__":
    main()
