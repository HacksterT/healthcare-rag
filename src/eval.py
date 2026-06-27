"""A lightweight self-evaluation: have the LLM rate groundedness and relevance.

Groundedness = is the answer supported by the retrieved context (not invented)?
Relevance    = does the answer actually address the question?
This is a cheap, honest signal, not a substitute for a human-graded test set.
"""

import json
import re

import ollama

from .config import LLM_MODEL, OLLAMA_HOST

_RUBRIC = (
    "Rate the assistant ANSWER on two axes, each 1-5 (5 best):\n"
    "- groundedness: is every claim supported by the CONTEXT?\n"
    "- relevance: does the answer address the QUESTION?\n"
    'Reply ONLY as JSON: {"groundedness": int, "relevance": int, "reason": "..."}'
)


def grade(query: str, result: dict, model: str | None = None) -> dict:
    model = model or LLM_MODEL
    context = "\n\n".join(f"[p. {h['page']}] {h['text']}" for h in result["hits"])
    msg = f"{_RUBRIC}\n\nQUESTION: {query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{result['answer']}"
    resp = ollama.Client(host=OLLAMA_HOST).chat(
        model=model, messages=[{"role": "user", "content": msg}]
    )
    text = resp["message"]["content"]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {"raw": text}
    except Exception:
        return {"raw": text}
