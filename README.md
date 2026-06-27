# Local Healthcare RAG

Retrieval-augmented question answering over a large medical reference manual, running
**entirely on local infrastructure**. Ollama provides both the embeddings
(`nomic-embed-text`) and the generation model; Chroma is the local vector store. No cloud,
no API keys, no patient data leaving the machine.

## What this demonstrates

- **Local-first RAG.** PDF → chunk → embed (nomic via Ollama) → Chroma → retrieve top-k →
  grounded, **page-cited** answer. The original coursework ran Mistral on a CUDA GPU and a
  separate embedding stack; this is rebuilt to use the Ollama server already on the machine.
- **Grounded and safe by construction.** The system prompt forces the model to answer only
  from retrieved context, cite page numbers, and refuse when the context lacks an answer.
- **Honest self-evaluation.** `src/eval.py` has the model rate its own answer for
  groundedness and relevance, a cheap signal (not a substitute for a human-graded test set).

## Prerequisites

[Ollama](https://ollama.com) running locally, with an embedding model and a chat model:

```bash
ollama pull nomic-embed-text                 # embeddings (required)
ollama pull gemma4:26b-a4b-it-q8_0           # generation (any chat model works)
ollama cp gemma4:26b-a4b-it-q8_0 gemma-q8    # stable alias the repo defaults to
```

The repo points at the alias `gemma-q8`, not a version-specific name, so upgrading the
model later is a one-liner (`ollama cp <new-model> gemma-q8`) with no code changes. Override
per-run with `RAG_LLM_MODEL` to test a different chat model.

A medical reference manual PDF is **not included** (copyright). Place your own at
`data/medical_diagnosis_manual.pdf`.

## Quickstart

```bash
uv sync

# Build the index (full manual; ~11 min for 4,000+ pages). Cap pages for a quick demo:
uv run python -m src.ingest --max-pages 500

# Query it
uv run python -c "from src.rag import answer; r=answer('What causes diabetes in pancreatic disease?'); print(r['sources']); print(r['answer'])"
```

Or open `notebooks/01-healthcare-rag.ipynb` for the full walkthrough (ingest, retrieve,
generate, self-evaluate).

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | local Ollama server |
| `RAG_EMBED_MODEL` | `nomic-embed-text:v1.5` | embedding model |
| `RAG_LLM_MODEL` | `gemma-q8` | generation model (a stable local Ollama alias) |

## Structure

```
healthcare-rag/
  data/medical_diagnosis_manual.pdf   (provide your own; gitignored)
  src/
    config.py    local endpoints + paths
    ingest.py    PDF -> chunks -> nomic embeddings -> Chroma  (CLI: python -m src.ingest)
    rag.py       retrieve() + grounded answer()
    eval.py      groundedness / relevance self-rating
  notebooks/01-healthcare-rag.ipynb
  chroma_db/     local persistent index (gitignored, rebuilt by ingest)
```

## Scope & safety

A **reference** tool for clinicians, grounded in a manual, not a source of individualized
medical advice. Answers are forced to cite their source pages and to decline when the manual
does not cover the question. Everything is local by design.

## Provenance

Originated as coursework (Mistral 7B on CUDA + LangChain + Chroma, with a Hugging Face
deployment). Rebuilt as a clean, fully local Ollama + Chroma system: configurable local
models, page-cited grounded answers, and self-evaluation. The copyrighted source manual is
not distributed; the work here is my own.
