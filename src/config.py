"""Local-first configuration. Everything points at the local Ollama server."""

import os
from pathlib import Path

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text:v1.5")
# Generation model: a stable local Ollama alias (see README). Re-point it at whatever
# model is current with `ollama cp <model> gemma-q8`; repos never need editing. Override
# per-run with RAG_LLM_MODEL (e.g. when testing a different chat model).
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "gemma-q8")

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "medical_diagnosis_manual.pdf"
CHROMA_DIR = str(ROOT / "chroma_db")
COLLECTION = "medical_manual"
