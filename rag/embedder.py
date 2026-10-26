"""Embed the text of chunks from md_chunker with the local embedding model."""
import os

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from core import utils  # noqa: F401  (importing it loads .env, so EMBED_MODEL_NAME / EMBED_DIM are available)
from rag.md_chunker import Chunk

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))

# Loaded once at import; normalized so cosine similarity works
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Embed every chunk's text (header + body) and store it in the chunk's `embedding` field (in place)."""
    vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError(f"Expected {len(chunks)} vectors, got {len(vectors)}")
    for chunk, vector in zip(chunks, vectors):
        if len(vector) != EMBED_DIM:
            raise ValueError(f"Expected a {EMBED_DIM}-dim embedding, got {len(vector)}")
        chunk.embedding = vector
    return chunks


def top_chunks(query: str, chunks: list[Chunk], k: int = 20) -> list[Chunk]:
    """Return the k chunks most similar to the query, best first (cosine similarity).

    chunks must come from embed_chunks, so that each one has its embedding set.
    """
    if not chunks:
        return []
    if any(chunk.embedding is None for chunk in chunks):
        raise ValueError("Some chunks have no embedding; run embed_chunks first")
    # All vectors are normalized, so the dot product is the cosine similarity
    scores = np.array([chunk.embedding for chunk in chunks]) @ np.array(embeddings.embed_query(query))
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]]
