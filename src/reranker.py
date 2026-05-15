"""Cross-encoder reranking utilities.

Dense retrieval is used for efficient candidate generation. A cross-encoder
reranker is then used to score query-chunk pairs more precisely.
"""

from __future__ import annotations

from typing import Any, Callable


def rerank_results(
    query: str,
    candidates: list[dict],
    reranker: Any,
    top_k: int = 5,
) -> list[dict]:
    """Rerank candidate chunks with a cross-encoder.

    Args:
        query: User query.
        candidates: Candidate retrieval results.
        reranker: CrossEncoder-like model with a predict method.
        top_k: Number of reranked results to return.

    Returns:
        Top-k reranked results. Each item includes rerank_score.
    """
    if not candidates:
        return []

    pairs = [(query, item["text"]) for item in candidates]
    rerank_scores = reranker.predict(pairs)

    for item, score in zip(candidates, rerank_scores):
        item["rerank_score"] = float(score)

    reranked = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    top_results = reranked[:top_k]

    for i, item in enumerate(top_results):
        item["rank"] = i + 1

    return top_results


def retrieve_with_rerank(
    query: str,
    retrieve_func: Callable,
    reranker: Any,
    first_k: int = 20,
    top_k: int = 5,
) -> list[dict]:
    """Run dense candidate retrieval followed by cross-encoder reranking.

    Args:
        query: User query.
        retrieve_func: Function with signature retrieve_func(query, top_k).
        reranker: CrossEncoder-like model.
        first_k: Number of candidates retrieved before reranking.
        top_k: Number of final reranked results.

    Returns:
        Top-k reranked results.
    """
    candidates = retrieve_func(query, top_k=first_k)
    return rerank_results(query=query, candidates=candidates, reranker=reranker, top_k=top_k)
