"""Sanity check: Python version, Groq chat model, embeddings, and vector store contents."""
import sys

from config import COLLECTION, EMBEDDING_MODEL, LLM_MODEL, QDRANT_PATH, get_embeddings, get_llm, get_qdrant

print(f"Python        : {sys.version.split()[0]}")
print(f"Chat model    : {LLM_MODEL} (Groq)")
print(f"Embeddings    : {EMBEDDING_MODEL} (local)")
print(f"Qdrant path   : {QDRANT_PATH}")

try:
    reply = get_llm().invoke("Reply with the single word: ok").content
    print(f"Groq call     : OK (replied: {str(reply).strip()[:40]})")
except Exception as exc:
    print(f"Groq call     : FAILED -> {exc}")
    sys.exit(1)

try:
    print(f"Embedding call: OK (dimension {len(get_embeddings().embed_query('test'))})")
except Exception as exc:
    print(f"Embedding call: FAILED -> {exc}")
    sys.exit(1)

client = get_qdrant()
if client.collection_exists(COLLECTION):
    print(f"Vector store  : OK ({client.count(COLLECTION).count} vectors in '{COLLECTION}')")
else:
    print("Vector store  : EMPTY -> run `python ingest.py`")