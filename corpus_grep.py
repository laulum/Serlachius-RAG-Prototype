# ---------------------------------------------------------------------------
# Serlachius RAG prototype: corpus search (corpus_grep.py)
#
# Part of the MBA thesis "Designing for Trustworthiness: An Action Design
# Research Study on AI Knowledge Retrieval System Redesign in a Cultural
# Institution" (Lauri Lumme, Haaga-Helia University of Applied Sciences, 2026).
#
# Used during the study to establish whether an answer exists in the
# corpus at all, for example to identify the answering passage for each
# question in the linguistic experiment (thesis, section 3.6).
# Comments were added on 19 November 2026 for publication. No code was
# changed: the code is as it was used in the study.
#
# Not published: the corpus (the Gösta Serlachius Fine Arts Foundation's
# publications), the vector store built from it, and the study's logs, which
# contain participants' questions. The interface and prompts are in Finnish;
# English translations are given in comments.
# ---------------------------------------------------------------------------
#
# It searches the chunk texts in the vector store with a regular expression,
# so it sees the text exactly as extracted and chunked. Exact matching misses
# inflected and compound Finnish forms; search with stems, as in the examples.
# Output: each hit with publication and page, then "N osumaa / M katkelmaa"
# ("N hits / M chunks").

"""
Does the corpus contain X at all? Read-only.

Distinguishes 'refused because the material isn't there' from
'refused because retrieval missed it'.

    python corpus_grep.py "kiina"
    python corpus_grep.py "kiina|shanghai|peking|kaukoit"
"""

import re
import sys
from pathlib import Path

import chromadb

# Paths assume the project lives in ~/serlachius-rag, as on the author's
# machine. Change them to run the code elsewhere.
DB_DIR     = Path.home() / "serlachius-rag" / "store"
COLLECTION = "serlachius_v4"


def main(pattern):
    db   = chromadb.PersistentClient(str(DB_DIR)).get_collection(COLLECTION)
    data = db.get(include=["documents", "metadatas"])
    rx   = re.compile(pattern, re.IGNORECASE)

    hits = 0
    for doc, meta in zip(data["documents"], data["metadatas"]):
        for m in rx.finditer(doc):
            hits += 1
            a, b = max(0, m.start() - 90), min(len(doc), m.end() + 90)
            print(f"{meta['doc']} s.{meta['page']}")
            print(f"   …{doc[a:b]}…\n")

    print(f"{hits} osumaa / {len(data['documents'])} katkelmaa  haulla /{pattern}/i")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python corpus_grep.py "<regex>"')
    main(sys.argv[1])