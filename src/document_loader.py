"""Document loading and preprocessing utilities.

This module handles:
1. PDF text extraction with page-level metadata.
2. Text chunking.
3. Cleaning noisy chunks such as table-of-contents pages, repeated headers,
   standalone page numbers, and short low-information chunks.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import fitz  # PyMuPDF
import pandas as pd


def extract_pdf_text(pdf_path: str | Path) -> list[dict]:
    """Extract page-level text from a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of dictionaries. Each dictionary contains:
        - page: 1-indexed PDF page number
        - text: extracted text from that page

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If the PDF contains no pages.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    if len(doc) == 0:
        raise ValueError(f"PDF file has no pages: {pdf_path}")

    pages: list[dict] = []

    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(
            {
                "page": page_idx + 1,
                "text": text,
            }
        )

    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150, min_chars: int = 100) -> list[str]:
    """Split text into overlapping character-based chunks.

    Args:
        text: Input text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.
        min_chars: Minimum number of characters required to keep a chunk.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If chunk_size <= overlap.
    """
    if text is None:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    chunks: list[str] = []
    start = 0
    text = str(text)

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if len(chunk) >= min_chars:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_chunks_dataframe(
    pages: Iterable[dict],
    source_name: str,
    chunk_size: int = 800,
    overlap: int = 150,
    min_chars: int = 100,
) -> pd.DataFrame:
    """Build a chunk-level dataframe from page-level text.

    Args:
        pages: Iterable of dictionaries containing page-level text.
        source_name: Human-readable document/source name.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters.
        min_chars: Minimum number of characters required to keep a chunk.

    Returns:
        DataFrame with columns:
        - source
        - page
        - chunk_id
        - text
    """
    all_chunks: list[dict] = []

    for page in pages:
        page_number = page.get("page")
        page_text = page.get("text", "")

        chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chars=min_chars,
        )

        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "source": source_name,
                    "page": page_number,
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

    return pd.DataFrame(all_chunks)


def clean_chunk_text(text: str) -> str:
    """Clean repeated headers, page numbers, whitespace, and PDF artifacts.

    Args:
        text: Raw chunk text.

    Returns:
        Cleaned chunk text.
    """
    if pd.isna(text):
        return ""

    lines = str(text).split("\n")
    cleaned_lines: list[str] = []

    noise_patterns = [
        r"^artificial intelligence index report 2025$",
        r"^artificial intelligence index report$",
        r"^artificial intelligence$",
        r"^index report 2025$",
        r"^stanford university$",
        r"^hai$",
        r"^chapter \d+$",
        r"^\d+$",
    ]

    for line in lines:
        line = line.strip()

        if not line:
            continue

        line_lower = line.lower()

        if any(re.fullmatch(pattern, line_lower) for pattern in noise_patterns):
            continue

        cleaned_lines.append(line)

    cleaned_text = " ".join(cleaned_lines)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def is_noisy_chunk(text: str, min_length: int = 200) -> bool:
    """Identify chunks that are likely not useful for retrieval.

    The function removes chunks that are too short, table-of-contents-like,
    or dominated by numbers and chapter listings.

    Args:
        text: Cleaned chunk text.
        min_length: Minimum length required to keep a chunk.

    Returns:
        True if the chunk should be removed, otherwise False.
    """
    if pd.isna(text):
        return True

    text = str(text).strip()
    text_lower = text.lower()

    if len(text) < min_length:
        return True

    if "table of contents" in text_lower:
        return True

    if text_lower.count("chapter") >= 5 and len(text_lower) < 1200:
        return True

    digit_tokens = re.findall(r"\b\d{1,4}\b", text_lower)
    words = re.findall(r"[a-zA-Z]+", text_lower)

    if len(words) > 0:
        digit_ratio = len(digit_tokens) / len(words)
        if digit_ratio > 0.35:
            return True

    return False


def clean_chunks_dataframe(
    chunks_df: pd.DataFrame,
    text_col: str = "text",
    clean_text_col: str = "clean_text",
    noisy_col: str = "is_noisy",
    min_length: int = 200,
) -> pd.DataFrame:
    """Clean and filter a chunk dataframe.

    Args:
        chunks_df: Raw chunks dataframe.
        text_col: Column containing raw chunk text.
        clean_text_col: Name of output cleaned-text column.
        noisy_col: Name of output noisy-flag column.
        min_length: Minimum length required to keep a cleaned chunk.

    Returns:
        Cleaned dataframe containing original metadata plus:
        - clean_text
        - is_noisy
    """
    required_columns = {"page", "chunk_id", text_col}
    missing = required_columns - set(chunks_df.columns)

    if missing:
        raise ValueError(f"Missing required columns in chunks_df: {sorted(missing)}")

    chunks_df_clean = chunks_df.copy()
    chunks_df_clean[clean_text_col] = chunks_df_clean[text_col].apply(clean_chunk_text)
    chunks_df_clean[noisy_col] = chunks_df_clean[clean_text_col].apply(
        lambda text: is_noisy_chunk(text, min_length=min_length)
    )

    chunks_df_clean = chunks_df_clean[~chunks_df_clean[noisy_col]].copy()
    chunks_df_clean = chunks_df_clean.reset_index(drop=True)

    return chunks_df_clean
