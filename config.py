"""Shared configuration: Groq chat model, local FastEmbed embeddings, embedded Qdrant."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)  # .env always wins over stale shell variables

COLLECTION = "enterprise_docs"
LLM_MODEL = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
QDRANT_PATH = str(BASE_DIR / "qdrant_data")  # always next to the code, whatever folder you run from


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX embeddings: no API key, no server."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key.startswith("gsk_your"):
        raise RuntimeError("GROQ_API_KEY is missing. Put your real key from console.groq.com/keys in .env")
    return ChatGroq(model=LLM_MODEL, temperature=0, api_key=key, max_retries=4, timeout=60)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return FastEmbedEmbeddings(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """One process-wide embedded Qdrant client (local mode locks the folder per process)."""
    return QdrantClient(path=QDRANT_PATH)