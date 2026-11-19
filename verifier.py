# ---------------------------------------------------------------------------
# Serlachius RAG prototype: year verifier, tiers 1 and 2 (verifier.py)
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
Post-hoc faithfulness verifier for the Serlachius RAG prototype.
Deterministic: no model calls, no training data, fully explainable.

Tier 1 (membership): does every year claimed in the answer appear in the retrieved sources?
Tier 2 (proximity):  for years that do appear, are they attached to the thing that was
                     asked about, or to a different fact?

Policy: three-tier PASS / WARNING / FAIL — flag, do not block.
"""

import re

# Years from 1500 to 2049, and Finnish decade expressions such as
# "1920-luvulla" ("in the 1920s").
YEAR_RE   = re.compile(r'\b(1[5-9]\d{2}|20[0-4]\d)\b')
DECADE_RE = re.compile(r'\b(1[5-9]\d0|20[0-4]0)-luvu\w*', re.IGNORECASE)

# Finnish question words and function words, never used as match terms.
STOPWORDS = {
    "milloin", "mitä", "miten", "missä", "kuka", "kenen", "mikä", "miksi", "kuinka",
    "ja", "tai", "sekä", "että", "on", "oli", "ovat", "hän", "hänen", "se", "sen",
    "kun", "jos", "myös", "vain", "ei", "eikä", "vaan", "kuten", "niin", "tämä", "tämän",
}

# Finnish case endings, longest first. Crude morphological approximation: no analyser
# dependency, but enough to reduce 'eron'->'ero', 'vaimostaan'->'vaimo', so that stems
# also match inside compounds ('avioeroa') and other inflected forms.
_SUFFIXES = ("staan","stään","ssaan","ssään","llaan","llään","ineen","hinsa","hinsä",
             "sta","stä","ssa","ssä","lla","llä","lle","lta","ltä","ksi","tta","ttä",
             "aan","ään","een","nsa","nsä","han","hän",
             "an","än","en","in","on","un","yn","na","nä","ta","tä",
             "a","ä","n","t","i")

def _destem(w, min_len=3):
    for suf in _SUFFIXES:
        if len(w) - len(suf) >= min_len and w.endswith(suf):
            return w[:-len(suf)]
    return w

def extract_years(text):
    """Explicit years, plus the base year of any decade expression (1920-luvulla -> 1920).
    Simplification: a decade claim is treated as a claim about its base year."""
    text = text or ""
    found = YEAR_RE.findall(text) + DECADE_RE.findall(text)
    return list(dict.fromkeys(found))

# How a year is judged to be "attached to the thing asked" (tier 2):
# each content word of the question is reduced to up to two 5-character
# prefixes (one from the word as written, one after stripping a case ending),
# so that inflected and compound forms still match. Prefixes that occur in
# more than 35 % of the SENTENCES of the retrieved text are dropped as
# non-discriminating, unless that would leave fewer than two, in which case
# the least frequent are kept. With fewer than ten sentences the share is not
# estimated and nothing is dropped.
# The 35 % threshold, the 5-character prefix and the 250-character window
# below were set by judgement during BIE 2 and were not tuned.
def question_keywords(question, source_blob="", min_len=4, stem_len=5, max_share=0.35):
    """
    Discriminative content-word stems from the question, for proximity matching.
    Stems that saturate the source are dropped: in a corpus about one person, that
    person's name sits beside every fact and discriminates nothing.
    """
    words = re.findall(r'\w+', question.lower())
    stems = []
    for w in words:
        if len(w) < min_len or w in STOPWORDS:
            continue
        for cand in {_destem(w)[:stem_len], w[:stem_len]}:
            if len(cand) >= 3 and cand not in stems:
                stems.append(cand)

    if not source_blob:
        return stems

    sentences = [s for s in re.split(r'[.!?]\s+', source_blob.lower()) if s.strip()]
    # With few sentences the share estimate is noise, and a genuinely topical stem
    # ('ero' in a passage that discusses a divorce twice) looks "saturating".
    if len(sentences) < 10:
        return stems

    shares = {s: sum(1 for sent in sentences if s in sent) / len(sentences) for s in stems}
    discriminative = [s for s in stems if shares[s] <= max_share]
    if len(discriminative) < 2:
        for s in sorted(stems, key=lambda x: shares[x]):
            if s not in discriminative:
                discriminative.append(s)
            if len(discriminative) >= 2:
                break
    return discriminative

# Text within 250 characters either side of each occurrence of the year.
def _windows_around(text, year, radius=250):
    out = []
    for m in re.finditer(r'\b' + re.escape(year) + r'\b', text):
        out.append(text[max(0, m.start()-radius):min(len(text), m.end()+radius)].lower())
    return out

# Statuses, as shown in the interface:
#   NO_CLAIMS  the answer states no year       "No years to check."
#   FAIL       a year in the answer is absent from every retrieved chunk (tier 1)
#              "Year(s) not found in the sources: ..."
#   WARNING    the year occurs in the sources, but no chunk has two question
#              terms within 250 characters of it (tier 2)
#              "Year ... appears in the source, but not in connection with the
#              thing asked; possible misattribution."
#   PASS       every year passes both tiers
#              "All years are found in the sources in connection with the
#              thing asked."
# Known limit: proximity cannot tell apart two events that share vocabulary
# (thesis, section 4.2.4). Names, places and other claims are not checked.
def verify(answer, retrieved_texts, question, proximity_radius=250):
    claimed = extract_years(answer)
    source_blob = "\n".join(retrieved_texts or [])
    source_years = set(extract_years(source_blob))

    if not claimed:
        return {"status": "NO_CLAIMS", "claimed_years": [], "fabricated": [],
                "unsupported_context": [], "verified": [],
                "note": "Ei tarkistettavia vuosilukuja."}

    fabricated, near, far = [], [], []
    stems = question_keywords(question, source_blob)

    for y in claimed:
        if y not in source_years:
            fabricated.append(y)
            continue
        # Require >=2 independent question terms in the window. Windows are computed
        # per chunk so a year in one chunk cannot be validated by words in another.
        need = 2 if len(stems) >= 2 else len(stems)
        hit = False
        for chunk in (retrieved_texts or []):
            for win in _windows_around(chunk, y, proximity_radius):
                if sum(1 for st in stems if st in win) >= need:
                    hit = True
                    break
            if hit:
                break
        (near if hit else far).append(y)

    if fabricated:
        status, note = "FAIL", f"Vuosiluku(j)a ei löydy lähteistä: {', '.join(fabricated)}."
    elif far:
        status = "WARNING"
        note = (f"Vuosiluku {', '.join(far)} esiintyy lähteessä, mutta ei kysytyn asian "
                f"yhteydessä — mahdollinen väärä kohdistus.")
    else:
        status, note = "PASS", "Kaikki vuosiluvut löytyvät lähteistä kysytyn asian yhteydestä."

    return {"status": status, "claimed_years": claimed, "fabricated": fabricated,
            "unsupported_context": far, "verified": near, "note": note}