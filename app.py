# ---------------------------------------------------------------------------
# Serlachius RAG prototype: web interface for the staff evaluation (app.py)
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
#
# Run with:  python app.py   then open http://127.0.0.1:5055
# Requires Ollama running locally with the poro2 model, and the vector store
# built by ingest_v4.py.
#
# Two modes share one interface (templates/index.html):
#   live mode      /            the participant types a question; ask.py runs
#   replay mode    /tapaus/<id> a prepared case is shown exactly as it was
#                               generated earlier, without calling the model
# Replay mode is how every participant saw identical output from a stochastic
# system (thesis, sections 3.5 and 5.4). Its cases come from tapaukset.json,
# which build_cases.py produces. That file is not published, because the
# source passages stored in it are text from the Foundation's publications.

"""
Staff-facing interface for the Serlachius RAG prototype (BIE 3).

This file WRAPS ask.py. It does not modify it. Retrieval, the prompt, the model,
the temperature and both verifier tiers are exactly as they stood at the end of
BIE 2: the artefact evaluated with staff is the artefact built and tested in
BIE 1-2.
Everything here is presentation and interaction logging.

Answers continue to be logged by ask.py to runlog.jsonl, untouched.
Interface events (sources opened, participant codes, timings) go to uilog.jsonl.
"""

import json
import httpx
import time
import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify

from ask import ask   # the frozen pipeline

# Paths assume the project lives in ~/serlachius-rag, as on the author's
# machine. Change them to run the code elsewhere.
CASES_PATH = Path.home() / "serlachius-rag" / "tapaukset.json"

# Four presentation orders of the ten prepared cases, one per participant
# (A-D in the thesis). The two Ukraine items (2, 9) are kept six to eight
# positions apart, and half of the participants meet each version first.
# /jarjestys/<key> lists one order as links for that participant.
ORDERS = {
    "A": [3, 2, 6, 7, 10, 5, 8, 1, 4, 9],
    "B": [5, 9, 4, 8, 1, 10, 7, 3, 6, 2],
    "C": [7, 4, 2, 5, 10, 3, 1, 6, 9, 8],
    "D": [6, 8, 9, 1, 4, 7, 5, 10, 3, 2],
}

# Interface events are appended to uilog.jsonl (not published). Event names:
#   istunto_alkoi   session started        kysymys         question asked
#   aikakatkaisu    timeout                virhe           error
#   tapaus_avattu   prepared case opened   lahde_avattu    source opened
#   viite_klikattu  in-text [n] marker clicked (sent from index.html)
UI_LOG = Path.home() / "serlachius-rag" / "uilog.jsonl"

app = Flask(__name__)

# Set at the start of each session so every event can be tied to a participant.
# The participant code is typed into the "Tunnus" (code) field of the interface.
SESSION = {"participant": ""}


def ui_log(event, **fields):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "participant": SESSION["participant"],
        "event": event,
    }
    entry.update(fields)
    with open(UI_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.get("/")
def index():
    return render_template("index.html", participant=SESSION["participant"])


# Route names are Finnish: osallistuja = participant, kysy = ask,
# tapahtuma = event, tapaus = case, jarjestys = order.
@app.post("/api/osallistuja")
def set_participant():
    SESSION["participant"] = (request.json or {}).get("participant", "").strip()
    ui_log("istunto_alkoi")
    return jsonify({"participant": SESSION["participant"]})


@app.post("/api/kysy")
def kysy():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Kysymys puuttuu."}), 400

    started = time.time()
    try:
        answer, hits, verdict, quote_verdict = ask(question)
    except httpx.ReadTimeout:
        ui_log("aikakatkaisu", question=question)
        return jsonify({"error": "Järjestelmä ei vastannut 90 sekunnissa."}), 504
    except Exception as exc:                      # Ollama down, model missing, etc.
        ui_log("virhe", question=question, error=str(exc))
        return jsonify({"error": f"Järjestelmä ei vastannut: {exc}"}), 500
    elapsed = round(time.time() - started, 1)

    sources = [
        {"n": i + 1, "doc": m["doc"], "page": m["page"], "text": t}
        for i, (t, m, _) in enumerate(hits)
    ]

    ui_log("kysymys", question=question, elapsed_s=elapsed,
           years_status=verdict["status"], quotes_status=quote_verdict["status"],
           n_sources=len(sources))

    return jsonify({
        "question": question,
        "answer": answer,
        "years":  {"status": verdict["status"],       "note": verdict["note"]},
        "quotes": {"status": quote_verdict["status"], "note": quote_verdict["note"]},
        "sources": sources,
        "elapsed_s": elapsed,
    })


@app.post("/api/tapahtuma")
def tapahtuma():
    data = request.json or {}
    ui_log(data.get("event", "tuntematon"),
           **{k: v for k, v in data.items() if k != "event"})
    return ("", 204)


# Replay mode: renders a stored case through the same template and the same
# display code as live answers. The language model is not called.
@app.get("/tapaus/<case_id>")
def tapaus(case_id):
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    case = next((c for c in cases if str(c["id"]) == case_id), None)
    if case is None:
        return f"Tuntematon tapaus: {case_id}", 404
    ui_log("tapaus_avattu", tapaus=case_id)
    return render_template("index.html",
                           participant=SESSION["participant"],
                           tapaus=case)


@app.get("/jarjestys/<key>")
def jarjestys(key):
    seq = ORDERS.get(key.upper())
    if not seq:
        return "Tuntematon järjestys", 404
    rows = "".join(f'<li><a href="/tapaus/{cid}">Kohde {pos}</a></li>'
                   for pos, cid in enumerate(seq, 1))
    return (f'<meta charset="utf-8"><h1>Järjestys {key.upper()}</h1>'
            f'<ol style="font:16px/2 sans-serif">{rows}</ol>')


if __name__ == "__main__":
    # Single user, local only. Threaded off so one question runs at a time.
    app.run(host="127.0.0.1", port=5055, debug=False, threaded=False)
