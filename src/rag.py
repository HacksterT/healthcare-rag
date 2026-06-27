"""Retrieval + grounded generation over the local Chroma store, using Ollama."""

import chromadb
import ollama

from .config import CHROMA_DIR, COLLECTION, EMBED_MODEL, LLM_MODEL, OLLAMA_HOST

SYSTEM = (
    "You are a clinical reference assistant. Answer ONLY from the provided context, which "
    "comes from a medical reference manual. Cite the page numbers you used, like [p. 123]. "
    "If the context does not contain the answer, say you do not have that information. This "
    "is reference information for clinicians, not individualized medical advice."
)


def _client():
    return ollama.Client(host=OLLAMA_HOST)


def model_available(name: str) -> bool:
    try:
        names = [m.model for m in _client().list().models]
        return name in names or f"{name}:latest" in names
    except Exception:
        return False


def retrieve(query: str, k: int = 4):
    client = _client()
    q_emb = client.embed(model=EMBED_MODEL, input=[query])["embeddings"][0]
    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
    res = col.query(query_embeddings=[q_emb], n_results=k)
    return [
        {"text": d, "page": m["page"], "distance": round(float(dist), 4)}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def answer(query: str, k: int = 4, model: str | None = None):
    model = model or LLM_MODEL
    hits = retrieve(query, k)
    sources = sorted({h["page"] for h in hits})
    if not model_available(model):
        return {
            "answer": None,
            "note": f"chat model '{model}' not found in Ollama; retrieval still works. "
            f"Pull one (e.g. `ollama pull llama3.2`) or set RAG_LLM_MODEL.",
            "sources": sources,
            "hits": hits,
        }
    context = "\n\n".join(f"[p. {h['page']}] {h['text']}" for h in hits)
    resp = _client().chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return {"answer": resp["message"]["content"], "sources": sources, "hits": hits}
