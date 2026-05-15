"""Dense retrieval utilities for RAG.

This module handles:
1. Embedding generation.
2. FAISS cosine-similarity index construction.
3. Top-k dense retrieval with page-level metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd


def build_embeddings(
    texts: list[str],
    embedding_model: Any,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Build sentence embeddings for a list of texts.

    Args:
        texts: List of strings to embed.
        embedding_model: SentenceTransformer-like model with an encode method.
        batch_size: Batch size for embedding generation.
        show_progress_bar: Whether to show a progress bar.

    Returns:
        Float32 numpy array of embeddings.
    """
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )

    return embeddings.astype("float32")


def build_faiss_cosine_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS cosine-similarity index.

    FAISS IndexFlatIP computes inner product. For L2-normalized vectors,
    inner product equals cosine similarity.

    Args:
        embeddings: Float32 embedding matrix of shape (n_chunks, dim).

    Returns:
        FAISS index.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D numpy array.")

    embeddings_cosine = embeddings.astype("float32").copy()
    faiss.normalize_L2(embeddings_cosine)

    dimension = embeddings_cosine.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_cosine)

    return index


def retrieve_dense_cosine(
    query: str,
    chunks_df: pd.DataFrame,
    index: faiss.Index,
    embedding_model: Any,
    text_col: str = "clean_text",
    top_k: int = 10,
) -> list[dict]:
    """Retrieve top-k chunks using cosine similarity.

    Args:
        query: User query.
        chunks_df: Chunk dataframe aligned with the FAISS index.
        index: FAISS index created from chunk embeddings.
        embedding_model: SentenceTransformer-like model.
        text_col: Name of the dataframe column containing retrieval text.
        top_k: Number of chunks to retrieve.

    Returns:
        List of retrieval result dictionaries.
    """
    if text_col not in chunks_df.columns:
        raise ValueError(f"Column '{text_col}' not found in chunks_df.")

    if index.ntotal != len(chunks_df):
        raise ValueError(
            f"FAISS index size ({index.ntotal}) does not match chunks_df length ({len(chunks_df)})."
        )

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results: list[dict] = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue

        row = chunks_df.iloc[idx]

        result = {
            "rank": rank + 1,
            "page": int(row["page"]),
            "chunk_id": int(row["chunk_id"]),
            "cosine_score": float(scores[0][rank]),
            "text": row[text_col],
        }

        if "source" in row:
            result["source"] = row["source"]

        results.append(result)

    return results


def save_faiss_index(index: faiss.Index, output_path: str | Path) -> None:
    """Save a FAISS index to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))


def load_faiss_index(index_path: str | Path) -> faiss.Index:
    """Load a FAISS index from disk."""
    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")

    return faiss.read_index(str(index_path))
