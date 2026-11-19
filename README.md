# Serlachius RAG prototype

A locally run, Finnish-language retrieval-augmented generation (RAG) prototype that answers staff questions from a museum's own publications, shows the passages each answer was generated from, and checks the years and quotations in the answer against those passages.

It was built and evaluated for the MBA thesis:

> Lumme, L. 2026. *Designing for Trustworthiness: An Action Design Research Study on AI Knowledge Retrieval System Redesign in a Cultural Institution.* Master's thesis. Haaga-Helia University of Applied Sciences.

The study was commissioned by the Gösta Serlachius Fine Arts Foundation (Mänttä, Finland). The prototype was developed over three build–intervention–evaluation cycles of Action Design Research and evaluated with the Foundation's staff in October 2026.

**Status.** This is a research prototype, published so that the study can be inspected. It is the code as used in the study, frozen for the staff evaluation, with explanatory comments added afterwards. It is not maintained and is not intended for production use.

## What it does

1. A staff member asks a question in Finnish.
2. The question is embedded with `intfloat/multilingual-e5-base`, and the four nearest passages are retrieved from a ChromaDB store built from the Foundation's publications.
3. A local language model (Poro 2, 8B, run through Ollama) answers from those passages only, citing them as [1]–[4]. If they do not contain a precise answer, it is instructed to refuse.
4. Two deterministic checks run on the answer. They flag problems; they never block or change the answer.
   - **Years** (tiers 1–2): is every year in the answer present in the passages, and does it occur near the terms of the question?
   - **Quotations** (tier 3): has a verbatim run of source text been altered inside the answer?
5. The interface shows the answer, the two check results and all four retrieved passages, with publication and page.

Everything runs on one machine. No query, passage or answer leaves it.

## Contents

| File | Role |
|---|---|
| `ask.py` | Retrieval, distance gate, prompt, generation, checks, logging |
| `verifier.py` | Year checks (tiers 1 and 2) |
| `quotecheck.py` | Quotation check (tier 3) |
| `app.py` | Web interface: live mode, and replay mode for the evaluation |
| `templates/index.html` | The interface |
| `ingest_v4.py` | Builds the vector store from the PDFs |
| `build_cases.py` | Builds the ten prepared evaluation cases from logged runs |
| `items.py` | The ten prepared cases (questions and the answers the prototype generated) |
| `corpus_grep.py` | Searches the stored text for a term, to establish whether an answer exists at all |

Each file begins with a comment explaining its role. The interface and prompts are in Finnish; English translations are given in the comments.

## What is not included, and why

- **The corpus** (29 of the Foundation's publications) and **the vector store** built from it: the publications are copyrighted, and the store contains their text in chunks.
- **The study's logs** (`runlog.jsonl`, `uilog.jsonl`): they contain the evaluation participants' questions and interface activity.
- **The prepared cases as displayed** (`tapaukset.json`): they include the retrieved source passages verbatim. `build_cases.py` shows how they were produced.

Without the corpus the code cannot reproduce the study's answers. It can be run on another set of PDFs.

## Configuration as evaluated

| Component | Setting |
|---|---|
| Corpus | 29 publications, 11,962 chunks (collection `serlachius_v4`) |
| Text extraction | `pdftotext` (poppler); endnote and bibliography pages excluded |
| Chunking | 800 characters, 150-character overlap, within a page |
| Embedding | `intfloat/multilingual-e5-base`, 768 dimensions, `query:` / `passage:` prefixes |
| Vector store | ChromaDB, squared L2 distance |
| Retrieval depth | k = 4 |
| Distance gate | threshold 1.0 on the nearest passage |
| Generation | Llama-Poro-2-8B-Instruct, Q4_K_M quantisation (GGUF), via Ollama; temperature 0.8; at most 400 tokens |
| Checks | Years (membership and proximity), quotations (altered verbatim runs) |

Chunk size, k, the distance threshold and the verifier's parameters were engineering defaults or judgement calls and were not tuned.

## Running it

Developed and evaluated on a MacBook with Apple M4, macOS, Python 3.13.12. In the study the prototype ran in the author's base Anaconda environment; `requirements.txt` pins the package versions installed there.

1. **Install Ollama** and register the language model under the name `poro2`. The study used Llama-Poro-2-8B-Instruct (Poro 2, 8B, by LumiOpen) at Q4_K_M quantisation, in the GGUF conversion published by `mradermacher` on Hugging Face:
   ```
   mkdir -p ~/models/poro2 && cd ~/models/poro2
   curl -L -o poro2-q4_k_m.gguf \
     "https://huggingface.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF/resolve/main/Llama-Poro-2-8B-Instruct.Q4_K_M.gguf"
   echo "FROM ./poro2-q4_k_m.gguf" > Modelfile
   ollama create poro2 -f Modelfile
   ```
   The model file used in the study has the SHA-256 checksum `ca30368561f9dd7636e8a36de2057f4deb729e395683a5cb1edea4263981d3e6` (check yours with `shasum -a 256 poro2-q4_k_m.gguf`). The model has its own licence; see its model card on Hugging Face.
2. **Install poppler**, which provides `pdftotext` (needed only for ingestion):
   `brew install poppler` (macOS) or `apt install poppler-utils` (Debian/Ubuntu).
3. **Create an environment and install the packages** (a separate environment is recommended):
   ```
   python3.13 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Adjust the paths.** The scripts expect the project in `~/serlachius-rag` and the PDFs in a folder set in `ingest_v4.py` (`MUSEUM`). Change these constants to match your layout.
5. **Build the store:** `python ingest_v4.py`
6. **Start the interface:** `python app.py`, then open <http://127.0.0.1:5055>.
   Or ask from the command line: `python ask.py` runs a set of test questions.

The embedding model is downloaded from Hugging Face on first use.

## Replay mode: how the evaluation was run

Because generation is stochastic (temperature 0.8), participants querying the live system would each have judged different answers. For the evaluation, ten answers the prototype had actually generated were fixed in advance and replayed through the same interface, without calling the model (`/tapaus/<id>`; the presentation orders are at `/jarjestys/A` to `/jarjestys/D`). The interface logged which sources each participant opened and in what order. The procedure is described in sections 3.5 and 5.4 of the thesis.

## Known limitations

- **The distance gate never fired in the study.** Observed distances to the nearest passage were 0.23–0.40 against a threshold of 1.0, so every refusal came from the prompt instruction.
- **The checks cover only years and verbatim quotations.** Names, places and the context of events are not checked. The proximity check cannot tell apart two events described in similar words.
- **Phrasing matters.** In the study's linguistic experiment, small changes in the wording of a question decided whether the answering passage was retrieved at all.
- **Page numbers.** The page number shown is the PDF page, which can differ from the printed edition.
- **Finnish only.** The prompt, the checks' word lists and the interface are Finnish.

## Licence

The code in this repository is released under the MIT License (see `LICENSE`). The licence covers only the code. It does not cover the Foundation's publications, the Poro 2 model or the Python packages the code depends on, which have their own terms.

## Acknowledgements

The staff of the Gösta Serlachius Fine Arts Foundation, who took part in the interviews and the evaluation, and the thesis supervisors Dr. Jorge Ropero Rodríguez and Dr. María García Antuña (Universidad de Sevilla) and Dr. Lili Aunimo (Haaga-Helia University of Applied Sciences).
