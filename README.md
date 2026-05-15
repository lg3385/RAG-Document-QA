# RAG-Based AI Report Question Answering System

A Retrieval-Augmented Generation (RAG) document question-answering system for long-form AI reports. The project processes a large PDF report, builds a cleaned retrieval corpus, performs semantic search with FAISS, improves ranking quality with a cross-encoder reranker, evaluates retrieval performance with page-aware metrics, and supports evidence-grounded answer generation.


## Project Overview

Large language models can generate fluent but unsupported answers when asked questions about long technical documents. This project addresses that problem by grounding answers in retrieved evidence from the source document.

The system first extracts text from a PDF, splits it into chunks with page-level metadata, removes noisy chunks such as table-of-contents pages and repeated headers, and builds a vector index for semantic retrieval. A cross-encoder reranker then reorders retrieved chunks to improve relevance. Finally, the system constructs evidence-based prompts for answer generation and evaluates retrieval quality using Recall@K, Precision@K, and MRR.

## Key Features

- PDF text extraction with page-level metadata
- Overlapping text chunking for long-document retrieval
- Noise filtering for table-of-contents chunks, repeated headers, standalone page numbers, and short low-information chunks
- Sentence embedding generation using `sentence-transformers`
- FAISS cosine-similarity vector retrieval
- Cross-encoder reranking for improved evidence ranking
- Page-aware retrieval benchmark using expected pages and evidence keywords
- Retrieval evaluation with Recall@K, Precision@K, and MRR
- Evidence-context construction for RAG answer generation
- Optional OpenAI-based answer generation
- No-API fallback mode that returns cited evidence snippets

## Methodology

### 1. Document Processing

The raw PDF is parsed with PyMuPDF. Each page is extracted as text and stored with its page number. The extracted text is then split into overlapping chunks so that the retrieval system can search over manageable text units while preserving page-level source information.

### 2. Corpus Cleaning

Initial retrieval experiments showed that table-of-contents chunks and repeated report headers can create false positives. To improve retrieval reliability, the system removes:

- Table of Contents chunks
- Repeated report headers
- Standalone page numbers
- Very short chunks
- Chunks dominated by chapter listings or numeric tokens

### 3. Dense Retrieval

Each cleaned chunk is embedded using a sentence-transformer model. The embeddings are normalized and indexed with FAISS. Cosine similarity is used for semantic search.

### 4. Cross-Encoder Reranking

Dense retrieval is used to retrieve candidate chunks efficiently. A cross-encoder reranker then scores each query-chunk pair and reorders the candidates to improve the quality of the top retrieved evidence.

### 5. Page-Aware Evaluation

The project uses a more realistic retrieval benchmark than keyword-only matching. A retrieved chunk is counted as relevant only if it satisfies both conditions:

1. The retrieved page is close to a manually labeled expected page.
2. The retrieved text contains expected evidence keywords.

This helps reduce false positives where a chunk contains a keyword but does not provide meaningful evidence.

### 6. Evidence-Grounded Answer Generation

The top reranked chunks are converted into an evidence context with source numbers, page numbers, and chunk IDs. The answer-generation prompt instructs the model to answer only from the provided context and cite source pages when possible.

If no LLM API key is available, the system falls back to an extractive answer mode that returns the most relevant cited evidence snippets.

## Results

In the retrieval evaluation, both dense retrieval and reranked retrieval achieved strong coverage, while reranking improved the quality and ordering of the retrieved evidence.

| Method | Recall@5 | Precision@5 | MRR@5 |
|---|---:|---:|---:|
| Clean Dense Retrieval | 1.00 | 0.60 | 0.87 |
| Clean Dense Retrieval + Cross-Encoder Reranking | 1.00 | 0.86 | 0.95 |

The results show that reranking preserved full Recall@5 while improving Precision@5 and MRR@5. This means the system still retrieved relevant evidence for every benchmark query, but reranking moved more relevant chunks into higher positions.

## Project Structure

```text
RAG-Document-QA-System/
│
├── notebooks/
│   └── RAG_AI_Index_QA.ipynb
│
├── data/
│   ├── raw/
│   │   └── Artificial_ Intelligence_Index_Report_2025.pdf
│   │
│   └── processed/
│       ├── chunks_clean_v4.csv
│       ├── evaluation_summary_v4.csv
│       ├── evaluation_comparison_v4.csv
│       └── clean_embeddings_v3.npy
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── retriever.py
│   ├── reranker.py
│   └── evaluator.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

If `faiss-cpu` has installation issues on Windows, install it with Conda:

```bash
conda install -c conda-forge faiss-cpu
```

## Usage

### 1. Add the PDF

Place the raw PDF file in:

```text
data/raw/Artificial_ Intelligence_Index_Report_2025.pdf
```

If your PDF has a different filename, update `PDF_PATH` in the notebook.

### 2. Run the Notebook

Open and run:

```text
notebooks/RAG_AI_Index_QA.ipynb
```

The notebook will perform:

```text
PDF Loading
→ Text Chunking
→ Noise Cleaning
→ Embedding Generation
→ FAISS Retrieval
→ Cross-Encoder Reranking
→ Retrieval Evaluation
→ Evidence Context Construction
→ RAG Answer Generation
```

### 3. Optional OpenAI Answer Generation

To enable LLM-based answer generation, set an OpenAI API key:

```bash
set OPENAI_API_KEY=your_api_key_here
```

On macOS or Linux:

```bash
export OPENAI_API_KEY=your_api_key_here
```

If no API key is provided, the notebook will automatically use the fallback evidence-output mode.

## Example Questions

The system can answer questions such as:

- How has AI investment changed in recent years?
- What does the report say about the cost of AI inference?
- How is AI being used in healthcare?
- What risks or challenges are associated with AI adoption?
- What does the report say about AI regulation?

## Main Modules

### `document_loader.py`

Handles PDF parsing, text extraction, chunking, and noisy-chunk filtering.

### `retriever.py`

Builds sentence embeddings, creates a FAISS cosine-similarity index, and performs dense retrieval.

### `reranker.py`

Applies cross-encoder reranking to reorder retrieved chunks.

### `evaluator.py`

Implements page-aware Recall@K, Precision@K, MRR@K, and per-query retrieval comparison.

## Technologies Used

- Python
- PyMuPDF
- Pandas
- NumPy
- Sentence-Transformers
- FAISS
- Cross-Encoder Reranking
- Jupyter Notebook
- Optional OpenAI API


## Future Improvements

- Add OCR for image-heavy PDFs
- Add multimodal figure and chart understanding
- Expand the evaluation set with more manually labeled questions
- Add hybrid retrieval with BM25 and dense embeddings
- Package the system as a command-line tool or Streamlit demo
- Add automated answer faithfulness evaluation
