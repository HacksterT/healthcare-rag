"""Ingest the medical manual into a local Chroma store using nomic embeddings (via Ollama)."""

import chromadb
import ollama
from pypdf import PdfReader

from .config import CHROMA_DIR, COLLECTION, EMBED_MODEL, OLLAMA_HOST, PDF_PATH


def chunk_text(text: str, size: int = 1000, overlap: int = 150):
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return out


def load_chunks(pdf_path=PDF_PATH, max_pages=None):
    reader = PdfReader(str(pdf_path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    chunks = []
    for i, page in enumerate(pages):
        for j, ch in enumerate(chunk_text(page.extract_text() or "")):
            if len(ch.strip()) > 50:
                chunks.append({"id": f"p{i + 1}_{j}", "text": ch, "page": i + 1})
    return chunks


def build_index(max_pages=None, batch_size=64, reset=True):
    client = ollama.Client(host=OLLAMA_HOST)
    chunks = load_chunks(max_pages=max_pages)
    store = chromadb.PersistentClient(path=CHROMA_DIR)
    if reset:
        try:
            store.delete_collection(COLLECTION)
        except Exception:
            pass
    col = store.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embs = client.embed(model=EMBED_MODEL, input=[c["text"] for c in batch])["embeddings"]
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embs,
            metadatas=[{"page": c["page"]} for c in batch],
        )
    return col.count()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the medical-manual Chroma index.")
    ap.add_argument("--max-pages", type=int, default=None, help="cap pages (default: all 4114)")
    n = build_index(max_pages=ap.parse_args().max_pages)
    print(f"indexed {n} chunks")
