# ---------------------------------------------------------------------------
# Serlachius RAG prototype: builder of the prepared evaluation cases (build_cases.py)
#
# Part of the MBA thesis "Designing for Trustworthiness: An Action Design
# Research Study on AI Knowledge Retrieval System Redesign in a Cultural
# Institution" (Lauri Lumme, Haaga-Helia University of Applied Sciences, 2026).
#
# This script built the ten prepared cases for the staff evaluation on
# 19 October 2026. It needs the study's runlog.jsonl and the vector store,
# neither of which is published, so it documents the procedure rather than
# running as published.
# Comments were added on 19 November 2026 for publication. No code was
# changed: the code is as it was used in the study.
#
# Not published: the corpus (the Gösta Serlachius Fine Arts Foundation's
# publications), the vector store built from it, and the study's logs, which
# contain participants' questions. The interface and prompts are in Finnish;
# English translations are given in comments.
# ---------------------------------------------------------------------------
#
# The procedure: each case pairs a question with one answer that the frozen
# prototype actually generated and logged. The script finds that run in the
# log, re-runs retrieval (which is deterministic) and refuses to build the
# case if the four retrieved sources differ from those logged at the time.
# The answer and both verification results are copied from the log unchanged,
# except for one marker format fix described below.

"""
Build tapaukset.json for the BIE 3 prepared-case renderer.

Each case is shaped exactly like the JSON /api/kysy returns, so index.html's
existing display functions render it unchanged — same CSS, same [n] regex,
same logging. Nothing that renders is modified.

  answer, years, quotes  <- runlog.jsonl, exactly as generated at the time
  sources (chunk texts)  <- re-retrieved; retrieval is deterministic
  question, id           <- the curated item sheet
  elapsed_s              <- uilog.jsonl where available (cosmetic only)

One deliberate edit: item 4's citation marker [1,2] becomes [1][2]. index.html
matches /\[(\d+)\]/g, so [1,2] never matched and rendered as inert text. No
claim changes. The fix is also noted in the thesis (Appendix 3).

(An earlier version read the item sheet with openpyxl; items.py replaced
that, and openpyxl is no longer needed.)
"""

import json
import re
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Paths assume the project lives in ~/serlachius-rag, as on the author's
# machine. Change them to run the code elsewhere.
HOME       = Path.home() / "serlachius-rag"
RUNLOG     = HOME / "runlog.jsonl"
UILOG      = HOME / "uilog.jsonl"
OUT        = HOME / "tapaukset.json"
DB_DIR     = HOME / "store"
COLLECTION = "serlachius_v4"
K          = 4

def read_sheet():
    """The curated items. See items.py for provenance."""
    from items import ITEMS
    return [{"id": i, "question": q, "answer": a} for i, q, a in ITEMS]


def read_runlog():
    """(question, answer) -> the most recent log entry for that exact pair."""
    index = {}
    with open(RUNLOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            index[(e["question"].strip(), e["answer"].strip())] = e
    return index


def read_elapsed():
    """question -> elapsed_s from the last matching UI event. Cosmetic field."""
    out = {}
    if not UILOG.exists():
        return out
    with open(UILOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("event") == "kysymys" and "elapsed_s" in e:
                out[e.get("question", "").strip()] = e["elapsed_s"]
    return out


def retrieve(emb, db, question, k=K):
    qv  = emb.encode([f"query: {question}"]).tolist()
    res = db.query(query_embeddings=qv, n_results=k,
                   include=["documents", "metadatas", "distances"])
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def main():
    items   = read_sheet()
    runlog  = read_runlog()
    elapsed = read_elapsed()

    print(f"{len(items)} items in sheet, {len(runlog)} unique runs in log\n")

    emb = SentenceTransformer("intfloat/multilingual-e5-base")
    db  = chromadb.PersistentClient(str(DB_DIR)).get_collection(COLLECTION)

    cases, problems = [], []

    for it in items:
        key   = (it["question"], it["answer"])
        entry = runlog.get(key)
        if entry is None:
            problems.append(f"item {it['id']}: no runlog entry matches this "
                            f"question+answer pair")
            continue

        hits = retrieve(emb, db, it["question"])

        # The log records which doc/page were retrieved. If re-retrieval differs,
        # something has changed since the run and the case would be wrong.
        logged = [(r["doc"], r["page"]) for r in entry["retrieved"]]
        fresh  = [(m["doc"], m["page"]) for _, m, _ in hits]
        if logged != fresh:
            problems.append(f"item {it['id']}: retrieval differs from the log\n"
                            f"    logged: {logged}\n    now:    {fresh}")
            continue

        answer = entry["answer"]
        if "[1,2]" in answer:                       # see module docstring
            answer = answer.replace("[1,2]", "[1][2]")
            print(f"item {it['id']}: normalised [1,2] -> [1][2]")

        cases.append({
            "id":       it["id"],
            "question": it["question"],
            "answer":   answer,
            "years":    {"status": entry["verification"]["status"],
                         "note":   entry["verification"]["note"]},
            "quotes":   {"status": entry["quote_check"]["status"],
                         "note":   entry["quote_check"]["note"]},
            "sources":  [{"n": i + 1, "doc": m["doc"], "page": m["page"], "text": t}
                         for i, (t, m, _) in enumerate(hits)],
            "elapsed_s": elapsed.get(it["question"], 5.0),
        })

    if problems:
        print("\n".join("PROBLEM  " + p for p in problems))
        sys.exit(f"\n{len(problems)} item(s) could not be built. Nothing written.")

    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"\nwrote {len(cases)} cases to {OUT}")
    for c in cases:
        miss = "" if c["question"] in elapsed else "   (elapsed approximated)"
        print(f"  {c['id']:>3}  {c['years']['status']:<10} "
              f"{c['quotes']['status']:<14} {len(c['sources'])} sources{miss}")


if __name__ == "__main__":
    main()