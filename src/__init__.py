"""RAG Document QA System.

Reusable modules for PDF loading, retrieval, reranking, and evaluation.

Main components:
- document_loader: PDF parsing, text chunking, and chunk cleaning.
- retriever: embedding generation, FAISS index construction, and dense retrieval.
- reranker: cross-encoder reranking.
- evaluator: page-aware retrieval evaluation metrics.
"""

from .document_loader import (
    extract_pdf_text,
    chunk_text,
    build_chunks_dataframe,
    clean_chunk_text,
    is_noisy_chunk,
    clean_chunks_dataframe,
)

from .retriever import (
    build_embeddings,
    build_faiss_cosine_index,
    retrieve_dense_cosine,
)

from .reranker import (
    rerank_results,
    retrieve_with_rerank,
)

from .evaluator import (
    page_hit,
    keyword_hit_min_matches,
    is_relevant_result,
    evaluate_recall_at_k_page_keyword,
    evaluate_precision_at_k_page_keyword,
    evaluate_mrr_at_k_page_keyword,
    build_retrieval_comparison,
)

__all__ = [
    "extract_pdf_text",
    "chunk_text",
    "build_chunks_dataframe",
    "clean_chunk_text",
    "is_noisy_chunk",
    "clean_chunks_dataframe",
    "build_embeddings",
    "build_faiss_cosine_index",
    "retrieve_dense_cosine",
    "rerank_results",
    "retrieve_with_rerank",
    "page_hit",
    "keyword_hit_min_matches",
    "is_relevant_result",
    "evaluate_recall_at_k_page_keyword",
    "evaluate_precision_at_k_page_keyword",
    "evaluate_mrr_at_k_page_keyword",
    "build_retrieval_comparison",
]
