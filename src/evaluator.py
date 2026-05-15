"""Evaluation utilities for document retrieval.

This module implements page-aware evaluation for RAG retrieval systems.

A retrieved result is considered relevant only when it satisfies both:
1. It is close to a manually labeled expected page.
2. It contains expected evidence keywords.

Metrics:
- Recall@K
- Precision@K
- MRR@K
- Per-query comparison between retrieval methods
"""

from __future__ import annotations

from typing import Callable

import pandas as pd


def page_hit(result_page: int, expected_pages: list[int], tolerance: int = 1) -> bool:
    """Check whether result_page is close to any expected page.

    Args:
        result_page: Retrieved page number.
        expected_pages: Manually labeled relevant pages.
        tolerance: Allowed page-distance tolerance.

    Returns:
        True if page is within tolerance, otherwise False.
    """
    for expected_page in expected_pages:
        if abs(result_page - expected_page) <= tolerance:
            return True

    return False


def keyword_hit_min_matches(
    text: str,
    keywords: list[str],
    min_matches: int = 1,
) -> tuple[bool, list[str]]:
    """Check whether text contains at least min_matches keywords.

    Args:
        text: Retrieved text.
        keywords: Expected evidence keywords.
        min_matches: Minimum number of keyword matches required.

    Returns:
        Tuple:
        - hit flag
        - matched keywords
    """
    text_lower = str(text).lower()
    matched: list[str] = []

    for keyword in keywords:
        if str(keyword).lower() in text_lower:
            matched.append(keyword)

    return len(matched) >= min_matches, matched


def is_relevant_result(
    result: dict,
    expected_pages: list[int],
    expected_keywords: list[str],
    tolerance: int = 1,
    min_keyword_matches: int = 1,
) -> tuple[bool, list[str]]:
    """Evaluate whether a retrieval result is relevant.

    A result is relevant only if both page and keyword conditions are met.

    Args:
        result: Retrieval result dictionary containing page and text.
        expected_pages: Manually labeled relevant pages.
        expected_keywords: Expected evidence keywords.
        tolerance: Page matching tolerance.
        min_keyword_matches: Minimum number of keyword matches.

    Returns:
        Tuple:
        - relevance flag
        - matched keywords
    """
    page_ok = page_hit(
        result_page=int(result["page"]),
        expected_pages=expected_pages,
        tolerance=tolerance,
    )

    keyword_ok, matched_keywords = keyword_hit_min_matches(
        text=result["text"],
        keywords=expected_keywords,
        min_matches=min_keyword_matches,
    )

    return page_ok and keyword_ok, matched_keywords


def evaluate_recall_at_k_page_keyword(
    eval_df: pd.DataFrame,
    retrieve_func: Callable,
    k: int = 5,
    tolerance: int = 1,
    min_keyword_matches: int = 1,
) -> tuple[float, pd.DataFrame]:
    """Evaluate Recall@K using page + keyword relevance.

    Recall@K is counted as 1 for a query if at least one relevant result
    appears in the top K.

    Required eval_df columns:
    - question
    - expected_pages
    - expected_keywords

    Args:
        eval_df: Evaluation dataframe.
        retrieve_func: Function with signature retrieve_func(query, top_k).
        k: Retrieval cutoff.
        tolerance: Page matching tolerance.
        min_keyword_matches: Minimum keyword matches.

    Returns:
        Tuple of mean recall and per-query detail dataframe.
    """
    hits = 0
    rows: list[dict] = []

    for _, row in eval_df.iterrows():
        query = row["question"]
        expected_pages = row["expected_pages"]
        expected_keywords = row["expected_keywords"]

        results = retrieve_func(query, top_k=k)

        relevant_found = False
        hit_pages: list[int] = []
        matched_all: list[str] = []

        for result in results:
            relevant, matched = is_relevant_result(
                result=result,
                expected_pages=expected_pages,
                expected_keywords=expected_keywords,
                tolerance=tolerance,
                min_keyword_matches=min_keyword_matches,
            )

            if relevant:
                relevant_found = True
                hit_pages.append(result["page"])
                matched_all.extend(matched)

        if relevant_found:
            hits += 1

        rows.append(
            {
                "question": query,
                "recall_hit": int(relevant_found),
                "hit_pages": hit_pages,
                "matched_keywords": sorted(set(matched_all)),
            }
        )

    recall = hits / len(eval_df) if len(eval_df) else 0.0

    return recall, pd.DataFrame(rows)


def evaluate_precision_at_k_page_keyword(
    eval_df: pd.DataFrame,
    retrieve_func: Callable,
    k: int = 5,
    tolerance: int = 1,
    min_keyword_matches: int = 1,
) -> tuple[float, pd.DataFrame]:
    """Evaluate Precision@K using page + keyword relevance.

    Precision@K is the fraction of the top K results that are relevant,
    averaged across all queries.

    Args:
        eval_df: Evaluation dataframe.
        retrieve_func: Function with signature retrieve_func(query, top_k).
        k: Retrieval cutoff.
        tolerance: Page matching tolerance.
        min_keyword_matches: Minimum keyword matches.

    Returns:
        Tuple of mean Precision@K and per-query detail dataframe.
    """
    precision_scores: list[float] = []
    rows: list[dict] = []

    for _, row in eval_df.iterrows():
        query = row["question"]
        expected_pages = row["expected_pages"]
        expected_keywords = row["expected_keywords"]

        results = retrieve_func(query, top_k=k)

        relevant_count = 0
        relevant_pages: list[int] = []

        for result in results:
            relevant, _ = is_relevant_result(
                result=result,
                expected_pages=expected_pages,
                expected_keywords=expected_keywords,
                tolerance=tolerance,
                min_keyword_matches=min_keyword_matches,
            )

            if relevant:
                relevant_count += 1
                relevant_pages.append(result["page"])

        precision = relevant_count / k if k else 0.0
        precision_scores.append(precision)

        rows.append(
            {
                "question": query,
                "precision_at_k": precision,
                "relevant_count": relevant_count,
                "relevant_pages": relevant_pages,
            }
        )

    mean_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    return mean_precision, pd.DataFrame(rows)


def evaluate_mrr_at_k_page_keyword(
    eval_df: pd.DataFrame,
    retrieve_func: Callable,
    k: int = 5,
    tolerance: int = 1,
    min_keyword_matches: int = 1,
) -> tuple[float, pd.DataFrame]:
    """Evaluate MRR@K using page + keyword relevance.

    Args:
        eval_df: Evaluation dataframe.
        retrieve_func: Function with signature retrieve_func(query, top_k).
        k: Retrieval cutoff.
        tolerance: Page matching tolerance.
        min_keyword_matches: Minimum keyword matches.

    Returns:
        Tuple of MRR@K and per-query detail dataframe.
    """
    reciprocal_ranks: list[float] = []
    rows: list[dict] = []

    for _, row in eval_df.iterrows():
        query = row["question"]
        expected_pages = row["expected_pages"]
        expected_keywords = row["expected_keywords"]

        results = retrieve_func(query, top_k=k)

        first_hit_rank = None
        hit_page = None
        matched_keywords: list[str] = []
        hit_preview = None

        for i, result in enumerate(results):
            relevant, matched = is_relevant_result(
                result=result,
                expected_pages=expected_pages,
                expected_keywords=expected_keywords,
                tolerance=tolerance,
                min_keyword_matches=min_keyword_matches,
            )

            if relevant:
                first_hit_rank = i + 1
                hit_page = result["page"]
                matched_keywords = matched
                hit_preview = str(result["text"])[:300].replace("\n", " ")
                break

        reciprocal_rank = 0.0 if first_hit_rank is None else 1 / first_hit_rank
        reciprocal_ranks.append(reciprocal_rank)

        rows.append(
            {
                "question": query,
                "first_hit_rank": first_hit_rank,
                "reciprocal_rank": reciprocal_rank,
                "hit_page": hit_page,
                "matched_keywords": matched_keywords,
                "hit_preview": hit_preview,
            }
        )

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return mrr, pd.DataFrame(rows)


def build_retrieval_comparison(
    dense_mrr_detail: pd.DataFrame,
    rerank_mrr_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-query comparison table for dense vs reranked retrieval.

    Args:
        dense_mrr_detail: Detail dataframe from evaluate_mrr_at_k_page_keyword.
        rerank_mrr_detail: Detail dataframe from evaluate_mrr_at_k_page_keyword.

    Returns:
        Comparison dataframe with MRR change per query.
    """
    required_cols = [
        "question",
        "first_hit_rank",
        "reciprocal_rank",
        "hit_page",
        "matched_keywords",
        "hit_preview",
    ]

    dense_detail = dense_mrr_detail[required_cols].copy()
    rerank_detail = rerank_mrr_detail[required_cols].copy()

    comparison = dense_detail.merge(
        rerank_detail,
        on="question",
        suffixes=("_dense", "_rerank"),
    )

    comparison["mrr_change"] = (
        comparison["reciprocal_rank_rerank"] -
        comparison["reciprocal_rank_dense"]
    )

    return comparison
