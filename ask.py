# ---------------------------------------------------------------------------
# Serlachius RAG prototype: retrieval and generation pipeline (ask.py)
#
# Part of the MBA thesis "Designing for Trustworthiness: An Action Design
# Research Study on AI Knowledge Retrieval System Redesign in a Cultural
# Institution" (Lauri Lumme, Haaga-Helia University of Applied Sciences, 2026).
#
# This is the code of the prototype as frozen for the staff evaluation
# (BIE 3, sessions held on 21 October 2026).
# Comments were added on 19 November 2026 for publication. The only other
# change: two descriptive labels in the test questions at the end of the file
# were reworded to remove a participant code. The code is otherwise as it was
# used in the study.
#
# Not published: the corpus (the Gösta Serlachius Fine Arts Foundation's
# publications), the vector store built from it, and the study's logs, which
# contain participants' questions. The interface and prompts are in Finnish;
# English translations are given in comments.
# ---------------------------------------------------------------------------
#
# What one call to ask() does:
#   1. embeds the question with multilingual-e5-base ("query: " prefix)
#   2. retrieves the k = 4 nearest chunks from ChromaDB (squared L2 distance)
#   3. applies a distance gate (refuse without calling the model if even the
#      nearest chunk is too far away)
#   4. sends the numbered chunks and the question to Poro 2 through Ollama
#   5. runs the two post-hoc checks on the answer (verifier.py: years;
#      quotecheck.py: quotations); the checks flag, they never block
#   6. appends the run to runlog.jsonl
# Run directly (python ask.py) it answers the regression questions at the end.

import json
import datetime
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from pathlib import Path
from verifier import verify
from quotecheck import verify_quotes
from ollama import Client

# Paths assume the project lives in ~/serlachius-rag, as on the author's
# machine. Change them to run the code elsewhere.
DB_DIR = Path.home() / "serlachius-rag" / "store"
LOG_PATH = Path.home() / "serlachius-rag" / "runlog.jsonl"

# E5 embedding models expect the prefix 'query: ' on questions and
# 'passage: ' on documents (the latter is added in ingest_v4.py).
emb = SentenceTransformer("intfloat/multilingual-e5-base")
db = chromadb.PersistentClient(str(DB_DIR)).get_or_create_collection("serlachius_v4")
# ollama 0.6.x defaults to no timeout: a stalled generation waits forever.
llm = Client(timeout=90)

# Parameters as used in the study:
#   k = 4          number of chunks retrieved; an engineering default, not tuned
#   threshold = 1.0  distance gate on the nearest chunk (squared L2, smaller is
#                  closer); see the comment at the gate below
#   model = "poro2"  Poro 2, 8B parameters, Q4_K_M quantisation, served locally
#                  by Ollama. No data leaves the machine.
# Returns the answer, all k retrieved chunks with metadata and distances, and
# the two verification results.
def ask(question, k=4, threshold=1.0, model="poro2"):
    q_vec = emb.encode([f"query: {question}"]).tolist()
    results = db.query(query_embeddings=q_vec, n_results=k,
                        include=["documents", "metadatas", "distances"])
    hits = list(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]))

    # Deterministic refusal gate: if even the nearest chunk is further away than
    # the threshold, refuse without calling the language model. In the
    # evaluation this gate never fired: observed nearest distances fell between
    # 0.23 and 0.40, so every refusal in the study came from the prompt
    # instruction below, not from this gate (thesis, section 4.3.5).
    # Refusal text: "This cannot be found in the available materials."
    if not hits or min(d for _, _, d in hits) > threshold:
        answer = "Tätä ei löydy käytettävissä olevista aineistoista."
        used_hits = []
    else:
        used_hits = hits
        # The chunks are numbered [1]..[4] in rank order, with publication and
        # page. The interface lists the sources in the same order, so the
        # model's [n] markers point to the right entry.
        context = "\n\n".join(
            f"[{i+1}] ({m['doc']}, s. {m['page']}) {t}"
            for i, (t, m, _) in enumerate(used_hits)
        )
        # The prompt, in English translation:
        #   Answer the user's question ONLY on the basis of the material below.
        #   Do not use other information and do not guess. Mark every claim with
        #   a source reference [n] immediately after the claim. First check
        #   whether the material answers EXACTLY the thing asked (for example an
        #   exact date, year or reason); it is not enough that the material
        #   deals with the same TOPIC more generally. If the material only
        #   touches on the topic but does not contain a precise answer to the
        #   question asked, this counts as the SAME as no answer being found.
        #   If and only if the material does not contain a PRECISE answer to the
        #   question, answer EXACTLY as follows and nothing else: 'This cannot
        #   be found in the available materials.' Do not give partial answers,
        #   guesses, hints of uncertain information, or near-topic answers that
        #   do not directly answer the question asked. If the answer is found,
        #   answer concisely, in 2-4 sentences.
        # This is the third version of the prompt; versions 1 and 2 and what
        # each changed are described in the thesis (section 4.2.4).
        prompt = (
            "Vastaa käyttäjän kysymykseen VAIN alla olevan aineiston perusteella. "
            "Älä käytä muuta tietoa äläkä arvaa. "
            "Merkitse jokainen väite lähdeviitteellä [n] heti väitteen perässä. "
            "Tarkista ensin, vastaako aineisto TÄSMÄLLEEN kysyttyyn asiaan "
            "(esimerkiksi tarkkaan päivämäärään, vuoteen tai syyhyn) - ei riitä, "
            "että aineisto käsittelee samaa AIHETTA yleisemmin. "
            "Jos aineisto vain sivuaa aihetta mutta ei sisällä täsmällistä vastausta "
            "kysyttyyn kysymykseen, tämä lasketaan SAMAKSI kuin ettei vastausta löydy. "
            "Jos ja vain jos aineisto ei sisällä TÄSMÄLLISTÄ vastausta kysymykseen, "
            "vastaa TÄSMÄLLEEN näin äläkä mitään muuta: "
            "'Tätä ei löydy käytettävissä olevista aineistoista.' "
            "Älä anna osittaisia vastauksia, arvauksia, vihjeitä epävarmasta tiedosta, "
            "tai lähelle aihetta osuvia vastauksia jotka eivät suoraan vastaa kysyttyyn kysymykseen. "
            "Jos vastaus löytyy, vastaa tiiviisti, 2-4 lauseella.\n\n"
            f"AINEISTO:\n{context}\n\n"
            f"KYSYMYS: {question}"
        )
        # temperature 0.8 is Ollama's default. Generation is therefore
        # stochastic: the same question can produce different answers, which
        # is why the evaluation replayed logged runs (see app.py) and the
        # linguistic experiment generated each condition five times.
        # num_predict = 400 caps the answer length; it was added in BIE 3
        # after generations that ran on without ending.
        response = llm.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.8, "num_predict": 400},
        )  # To change temp: options={"temperature": 0, "num_predict": 400}
        answer = response["response"]
        if response.done_reason == "length":
            # "[The answer was cut off at the length limit.]"
            answer += "\n\n[Vastaus katkaistiin pituusrajan vuoksi.]"

    # --- post-hoc verification ---
    # Both checks see only the chunks that were given to the model. They report
    # a status and a note that the interface displays; they never change or
    # withhold the answer (flag, do not block).
    verdict = verify(answer, [t for t, _, _ in used_hits], question)
    quote_verdict = verify_quotes(answer, [t for t, _, _ in used_hits])

    # One JSON line per run. The study's own log is not published, because it
    # contains participants' questions.
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "question": question,
        "model": model,
        "threshold": threshold,
        "retrieved": [{"doc": m["doc"], "page": m["page"], "distance": d}
                      for _, m, d in hits],
        "used_for_answer": len(used_hits) > 0,
        "answer": answer,
        "verification": verdict,
        "quote_check": quote_verdict
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # Note: all k retrieved chunks are returned even when the answer is a
    # refusal, so the interface shows four sources under a refusal too.
    # Participants noticed this during the evaluation (thesis, section 4.3.4).
    return answer, hits, verdict, quote_verdict


if __name__ == "__main__":
    # Regression questions used during development (BIE 2), with notes on the
    # expected behaviour. In English: when did Gösta Serlachius divorce his
    # wife Sissi; how did he meet Ester Toivonen; what can you tell me about
    # his trip to Ukraine in 1918; when was he admitted to an isolation
    # hospital and why; when did he visit China; when are the next
    # parliamentary elections in Finland. The Ukraine and hospital questions
    # are false refusals of the existing system reported in the diagnostic
    # interviews.
    test_questions = [
        ("Milloin Gösta Serlachius otti eron vaimostaan Sissistä?", "date stated indirectly (May 1918, decision announced)"),
        ("Miten Gösta Serlachius tapasi Ester Toivosen?", "in-corpus, clean answer expected (regression check)"),
        ("Mitä voit kertoa Gösta Serlachiuksen tekemästä matkasta Ukrainaan 1918?", "false refusal reported in the interviews"),
        ("Milloin Gösta Serlachius joutui kulkutautisairaalaan ja miksi näin kävi?", "false refusal reported in the interviews"),
        ("Milloin Gösta Serlachius vieraili Kiinassa?", "topically adjacent but absent (soft refusal test)"),
        ("Milloin seuraavat eduskuntavaalit järjestetään Suomessa?", "fully out-of-corpus (threshold calibration)"),
    ]

    for q, note in test_questions:
        print(f"\n{'='*70}\nKYSYMYS: {q}\n[{note}]")
        answer, hits, verdict, quote_verdict = ask(q)
        print(f"VASTAUS: {answer}")
        print(f"TARKISTUS: [{verdict['status']}] {verdict['note']}")
        print(f"SITAATIT:  [{quote_verdict['status']}] {quote_verdict['note']}")
        print("LÄHTEET (etäisyys = pienempi on lähempänä):")
        for text, meta, dist in hits:
            print(f"  - {meta['doc']}, s.{meta['page']} (etäisyys={dist:.3f})")