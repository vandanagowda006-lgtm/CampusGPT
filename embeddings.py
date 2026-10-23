"""Batched, cached Gemini embeddings.
 
create_embeddings(texts, task_type) -> float32 array (n, dim), L2-normalised.
Results are cached in a small SQLite file, so re-uploading the same PDF costs
no API calls and never hits rate limits twice.
"""
 
import hashlib
import os
import random
import sqlite3
import time
from pathlib import Path
 
import numpy as np
from dotenv import load_dotenv
 
load_dotenv()
 
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CACHE_PATH = Path(__file__).resolve().parent / "embedding_cache" / "embeddings_v2.sqlite3"
BATCH_SIZE = 50
MAX_CHARS = 8000
 
_RETRYABLE = ("503", "UNAVAILABLE", "500", "INTERNAL", "429",
              "RESOURCE_EXHAUSTED", "502", "BAD_GATEWAY", "504", "DEADLINE_EXCEEDED")
 
_client = None
 
 
def _get_client():
    global _client
    if _client is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        _client = genai.Client(api_key=key)
    return _client
 
 
def _db():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
    return conn
 
 
def _key(text, task_type):
    return hashlib.sha256(f"{EMBED_MODEL}|{task_type}|{text}".encode("utf-8")).hexdigest()
 
 
def _normalise(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
 
 
def _embed_batch(texts, task_type, retries=6):
    from google.genai import types
 
    for attempt in range(retries):
        try:
            response = _get_client().models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors = [e.values for e in response.embeddings]
            if len(vectors) != len(texts):
                raise RuntimeError("Embedding service returned a different number of vectors.")
            return vectors
        except Exception as exc:
            retryable = any(m in str(exc).upper() for m in _RETRYABLE)
            if not retryable or attempt == retries - 1:
                if retryable:
                    raise RuntimeError(
                        "Gemini is busy or the request limit was reached. "
                        "Please wait a minute and try again."
                    ) from exc
                raise
            wait = min(60, 2 ** (attempt + 1)) + random.uniform(0, 0.5)
            print(f"Embedding API busy, retrying in {wait:.1f}s ({attempt + 1}/{retries - 1})...")
            time.sleep(wait)
 
 
def create_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):
    texts = [(t or "").strip()[:MAX_CHARS] or "empty" for t in texts]
    if not texts:
        raise ValueError("No texts to embed.")
 
    keys = [_key(t, task_type) for t in texts]
    vectors = [None] * len(texts)
 
    conn = _db()
    try:
        cached = {}
        for start in range(0, len(keys), 500):
            part = keys[start:start + 500]
            rows = conn.execute(
                f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(part))})", part
            ).fetchall()
            cached.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
 
        missing = []
        for i, k in enumerate(keys):
            if k in cached:
                vectors[i] = cached[k]
            else:
                missing.append(i)
 
        for start in range(0, len(missing), BATCH_SIZE):
            idx = missing[start:start + BATCH_SIZE]
            new_vectors = _normalise(_embed_batch([texts[i] for i in idx], task_type))
            conn.executemany(
                "INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)",
                [(keys[i], new_vectors[j].tobytes()) for j, i in enumerate(idx)],
            )
            conn.commit()
            for j, i in enumerate(idx):
                vectors[i] = new_vectors[j]
            print(f"Embedded {min(start + BATCH_SIZE, len(missing))}/{len(missing)} new chunks")
    finally:
        conn.close()
 
    return _normalise(np.vstack(vectors))
 
 
def create_embedding(text, task_type="RETRIEVAL_QUERY"):
    """Single-text helper kept for backward compatibility."""
    return create_embeddings([text], task_type=task_type)[0]
 
