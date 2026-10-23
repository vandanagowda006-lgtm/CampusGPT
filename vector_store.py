"""Hybrid search using FAISS + BM25 with section-aware retrieval."""

import math
import re
from collections import Counter, defaultdict

import faiss
import numpy as np


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or",
    "is", "are", "was", "were", "be", "by", "with", "as", "at",
    "it", "this", "that", "what", "which", "who", "how", "me",
    "my", "give", "tell", "show", "please",
}


def tokenize(text):
    return [
        word
        for word in _TOKEN.findall((text or "").lower())
        if word not in _STOP
    ]


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.postings = defaultdict(list)
        self.doc_len = []

        for doc_id, text in enumerate(documents):
            tokens = tokenize(text)
            self.doc_len.append(len(tokens))

            for term, frequency in Counter(tokens).items():
                self.postings[term].append(
                    (doc_id, frequency)
                )

        self.n = len(documents)
        self.avg_len = (
            sum(self.doc_len) / self.n
            if self.n
            else 1.0
        )

    def scores(self, query):
        result = defaultdict(float)

        for term in set(tokenize(query)):
            posting = self.postings.get(term)

            if not posting:
                continue

            document_frequency = len(posting)

            idf = math.log(
                1
                + (
                    self.n - document_frequency + 0.5
                )
                / (document_frequency + 0.5)
            )

            for doc_id, frequency in posting:
                denominator = frequency + self.k1 * (
                    1 - self.b
                    + self.b
                    * self.doc_len[doc_id]
                    / max(self.avg_len, 1.0)
                )

                result[doc_id] += (
                    idf
                    * frequency
                    * (self.k1 + 1)
                    / denominator
                )

        return result


class HybridIndex:
    def __init__(self, faiss_index, bm25, section_ids):
        self.faiss = faiss_index
        self.bm25 = bm25
        self.section_ids = section_ids

    @property
    def ntotal(self):
        return self.faiss.ntotal


def create_vector_store(chunks, embeddings):
    """Create a FAISS index and BM25 search index."""

    vectors = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError(
            "Embeddings must be a non-empty 2D array."
        )

    if len(chunks) != len(vectors):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Embeddings contain invalid numeric values."
        )

    faiss.normalize_L2(vectors)

    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)

    section_ids = defaultdict(list)

    for chunk_id, chunk in enumerate(chunks):
        section_key = chunk.get("section_key")

        if section_key:
            section_ids[section_key].append(chunk_id)

    index = HybridIndex(
        faiss_index,
        BM25([chunk["text"] for chunk in chunks]),
        dict(section_ids),
    )

    print(f"Vector store created with {len(chunks)} chunks.")

    return index, chunks


def search_vector_store(
    index,
    chunks,
    query_text,
    query_embedding,
    top_k=12,
    section_keys=None,
):
    """
    Search using semantic similarity and BM25.

    If section_keys match indexed sections, return every chunk
    belonging to those sections in document order.

    Otherwise, use hybrid ranking to retrieve relevant chunks.
    """

    n = len(chunks)

    if not n or index.ntotal == 0:
        return []

    # --------------------------------------------------
    # 1. Find chunks belonging to requested sections.
    # --------------------------------------------------

    candidates = None

    if section_keys:
        matching_ids = set()

        for key in section_keys:
            matching_ids.update(
                index.section_ids.get(key, [])
            )

        if matching_ids:
            candidates = matching_ids

            print(
                "Section retrieval:",
                section_keys,
                "| Matching chunks:",
                len(candidates),
            )

    # --------------------------------------------------
    # 2. Validate query embedding.
    # --------------------------------------------------

    query = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(1, -1)

    if query.shape[1] != index.faiss.d:
        raise ValueError(
            f"Query embedding dimension ({query.shape[1]}) "
            f"does not match index dimension ({index.faiss.d})."
        )

    faiss.normalize_L2(query)

    # --------------------------------------------------
    # 3. Exhaustive section retrieval.
    # --------------------------------------------------

    if candidates is not None:
        return [
            {
                "chunk_id": chunk_id,
                "chunk": chunks[chunk_id],
                "score": 0.0,
            }
            for chunk_id in sorted(candidates)
        ]

    # --------------------------------------------------
    # 4. Hybrid retrieval for general questions.
    # --------------------------------------------------

    pool = min(
        n,
        max(top_k * 4, 50),
    )

    _, vector_ids = index.faiss.search(
        query,
        pool,
    )

    vector_rank = [
        int(chunk_id)
        for chunk_id in vector_ids[0]
        if chunk_id >= 0
    ]

    bm25_scores = index.bm25.scores(query_text)

    bm25_rank = [
        chunk_id
        for chunk_id, score in sorted(
            bm25_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if score > 0
    ][:pool]

    # Reciprocal rank fusion.
    fused = defaultdict(float)

    for rank, chunk_id in enumerate(vector_rank):
        fused[chunk_id] += 1.0 / (60 + rank + 1)

    for rank, chunk_id in enumerate(bm25_rank):
        fused[chunk_id] += 1.0 / (60 + rank + 1)

    selected_ids = [
        chunk_id
        for chunk_id, score in sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
    ]

    return [
        {
            "chunk_id": chunk_id,
            "chunk": chunks[chunk_id],
            "score": fused[chunk_id],
        }
        for chunk_id in selected_ids
    ]