# ---------------------------------------------------------------------------
# Serlachius RAG prototype: quotation check, tier 3 (quotecheck.py)
#
# Part of the MBA thesis "Designing for Trustworthiness: An Action Design
# Research Study on AI Knowledge Retrieval System Redesign in a Cultural
# Institution" (Lauri Lumme, Haaga-Helia University of Applied Sciences, 2026).
#
# This is the code of the prototype as frozen for the staff evaluation
# (BIE 3, sessions held on 21 October 2026).
# Comments were added on 19 November 2026 for publication. No code was
# changed: the code is as it was used in the study.
#
# Not published: the corpus (the Gösta Serlachius Fine Arts Foundation's
# publications), the vector store built from it, and the study's logs, which
# contain participants' questions. The interface and prompts are in Finnish;
# English translations are given in comments.
# ---------------------------------------------------------------------------

"""
Tier 3: altered-quotation detection.

Signature of a fabricated quote: a long run of words matching the source EXACTLY,
then one or two words that do not, then another long exact run. That pattern means the
model reproduced source text and changed something inside it — the answer survives a
citation check while misrepresenting what the source says.

Genuine paraphrase produces no long exact runs and is not flagged.
Exact reproduction produces one long run with no interruption and is not flagged.
"""
import re, difflib

# The comparison is word-level (difflib) between the answer and all retrieved
# chunks joined together. A "quote" here is any long verbatim run of source
# text, whether or not the answer puts it in quotation marks. The three
# thresholds below were set by judgement during BIE 2 and were not tuned.
MIN_FLANK   = 3    # exact-match words required on EACH side of the alteration
MIN_TOTAL   = 8    # ...and this many exact words across both sides combined
MAX_GAP     = 3    # longest altered stretch still counted as "a changed quote"

def _words(s):
    s = (s or "").lower().replace('\u201d', '"').replace('\u201c', '"')
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().split()

# Statuses, as shown in the interface:
#   NO_SOURCE      no chunks were given to the model (the distance gate in
#                  ask.py refused)                    "No source text."
#   QUOTE_ALTERED  a verbatim run has a short alteration inside it
#                  "A passage repeating the source has been altered:
#                   in the source "..." -> in the answer "..." "
#   QUOTES_OK      a verbatim run of 8+ words with no alteration
#                  "Verbatim quotation (n words) matches the source."
#   NO_QUOTES      no long verbatim run               "No direct quotations."
# "(poistettu)" = "(deleted)", "(lisätty)" = "(added)".
def verify_quotes(answer, retrieved_texts):
    a = _words(answer)
    b = _words("\n".join(retrieved_texts or []))
    if not a or not b:
        return {"status": "NO_SOURCE", "altered": [], "note": "Ei lähdetekstiä."}

    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = sm.get_opcodes()

    altered, longest_exact = [], 0
    for i, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal":
            longest_exact = max(longest_exact, i2 - i1)
            continue
        if tag not in ("replace", "delete", "insert"):
            continue
        if (i2 - i1) > MAX_GAP and (j2 - j1) > MAX_GAP:
            continue                                  # too big to be a tweak
        prev_ = ops[i-1] if i > 0 else None
        next_ = ops[i+1] if i + 1 < len(ops) else None
        lflank = (prev_[2] - prev_[1]) if prev_ and prev_[0] == "equal" else 0
        rflank = (next_[2] - next_[1]) if next_ and next_[0] == "equal" else 0
        if (lflank >= MIN_FLANK and rflank >= MIN_FLANK
                and lflank + rflank >= MIN_TOTAL):
            altered.append({
                "in_answer": " ".join(a[i1:i2]) or "(poistettu)",
                "in_source": " ".join(b[j1:j2]) or "(lisätty)",
                "context": " ".join(a[max(0, i1-6):i2+6]),
            })

    if altered:
        bits = "; ".join(f"lähteessä \u201d{x['in_source']}\u201d \u2192 vastauksessa \u201d{x['in_answer']}\u201d"
                         for x in altered)
        return {"status": "QUOTE_ALTERED", "altered": altered,
                "note": f"Lähdettä toistava jakso on muutettu: {bits}"}
    if longest_exact >= 8:
        return {"status": "QUOTES_OK", "altered": [],
                "note": f"Sanatarkka lainaus ({longest_exact} sanaa) vastaa lähdettä."}
    return {"status": "NO_QUOTES", "altered": [], "note": "Ei suoria lainauksia."}