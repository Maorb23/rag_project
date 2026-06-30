# FinanceBench RAG Pipeline

End-to-end Retrieval-Augmented Generation (RAG) pipeline for answering financial QA questions from the [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) benchmark. The project compares a no-context LLM baseline against a grounded RAG system built over source company PDFs, then evaluates answer correctness, faithfulness, and retrieval quality.

The implementation is intentionally modular: notebooks orchestrate experiments, while reusable pipeline logic lives under `src/financebench_rag`.

## What This Project Does

The pipeline:

1. Loads FinanceBench from Hugging Face.
2. Filters to `domain-relevant` and `novel-generated` questions.
3. Repairs or normalizes PDF links for referenced company filings.
4. Runs a naive LLM baseline without retrieved context.
5. Downloads the referenced PDFs, loads pages with metadata, chunks them, embeds them, and stores them in FAISS.
6. Retrieves relevant chunks for each question and generates context-grounded answers.
7. Compares naive vs. RAG answers.
8. Evaluates the RAG system with:
   - LLM-as-judge correctness
   - Ragas faithfulness on a limited subset
   - Page-level retrieval Hit@k

## Repository Layout

```text
.
+-- data/                         # Downloaded PDFs and local data artifacts
+-- results/                      # CSV/JSON outputs from pipeline stages
+-- src/financebench_rag/         # Reusable pipeline package
|   +-- config.py                 # Environment-driven pipeline configuration
|   +-- dataset.py                # FinanceBench loading, filtering, link repair
|   +-- vectorstore.py            # PDF loading, chunking, FAISS persistence
|   +-- rag_pipeline.py           # Retrieval and grounded answer generation
|   +-- naive_generation.py       # No-retrieval baseline
|   +-- evaluation.py             # Correctness, faithfulness, Hit@k metrics
|   +-- comparison.py             # Side-by-side answer comparison
|   +-- pipeline.py               # Full stage-by-stage orchestration
+-- vectorstore/                  # Persisted FAISS indexes
+-- pipeline_collab.ipynb         # Collaborative/working notebook
+-- Pipeline_notebook_7.ipynb     # Full pipeline notebook
+-- prompt.MD                     # Original project brief
+-- requirements.txt              # Python dependencies
```

## Core Design

### Dataset Preparation

`dataset.py` loads `PatronusAI/financebench`, keeps only the two target question types, normalizes evidence page numbers, and repairs dead document links using the canonical FinanceBench PDF URL pattern.

### Vector Store

`vectorstore.py` builds a document index from the filtered dataset's referenced PDFs only. PDFs are loaded with `PyPDFLoader`, one LangChain `Document` per page, and each page receives standardized metadata:

- `doc_name`
- `company`
- `doc_period`
- `page_number`

Pages are chunked with `RecursiveCharacterTextSplitter` using the configured `chunk_size` and `chunk_overlap`, embedded with `BAAI/bge-small-en-v1.5`, and persisted to FAISS for reuse.

### RAG Generation

`rag_pipeline.py` retrieves the top-k chunks, formats them with source metadata, and calls a Nebius-compatible OpenAI client. The system prompt instructs the model to answer only from retrieved context, cite source documents, and state when the context is insufficient.

Optional embedding-based reranking can retrieve a larger candidate set and rescore it before final generation.

### Evaluation

`evaluation.py` reports three complementary signals:

- **Correctness**: an LLM judge compares the RAG answer against the ground truth.
- **Faithfulness**: Ragas scores whether the answer is supported by retrieved context.
- **Page Hit@k**: retrieval succeeds if any returned chunk page matches any ground-truth evidence page.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Because this repo uses a `src/` layout without a packaging file, expose the source directory before running scripts from the terminal:

```powershell
$env:PYTHONPATH = "src"
```

## Configuration

Create a local `.env` file in the repo root. Do not commit real API keys.

```env
FINANCEBENCH_DATASET_ID=PatronusAI/financebench
FINANCEBENCH_DATASET_SPLIT=train

NEBIUS_BASE_URL=https://api.studio.nebius.com/v1
NEBIUS_API_KEY=your_api_key_here

GENERATION_MODEL=your_generation_model
JUDGE_MODEL=your_judge_model
RAGAS_MODEL=your_ragas_model

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_DEFAULT_K=4
RETRIEVAL_HIT_K_VALUES=1,3,5

RERANK_ENABLED=false
RERANK_TOP_K=20
RERANK_FINAL_K=4

DATA_DIR=data
PDF_DIR=data/pdfs
VECTORSTORE_DIR=vectorstore
RESULTS_DIR=results
```

## Running The Pipeline

The full pipeline can be run from a notebook, or directly from Python:

```powershell
$env:PYTHONPATH = "src"
python -c "from financebench_rag.config import load_config; from financebench_rag.pipeline import execute_full_pipeline; execute_full_pipeline(load_config())"
```

On the first run, the pipeline downloads required PDFs and builds the FAISS index. Later runs reuse the persisted vector store when `index.faiss` is present.

## Main Outputs

Pipeline outputs are written to `results/` as CSV and JSON files. Important artifacts include:

```text
stage1_filtered.csv/json            # Filtered FinanceBench questions
stage1_doc_mapping.csv/json         # Repaired document URL mapping
stage2_naive.csv/json               # No-retrieval baseline answers
stage3_sanity_checks.csv/json       # Retrieval spot checks
stage4_rag_stage2_questions.csv/json # RAG answers for sampled questions
stage5_comparison.csv/json          # Ground truth vs naive vs RAG
stage6_rag_full.csv/json            # RAG answers for full filtered set
stage6_correctness.csv/json         # LLM judge results
stage6_faithfulness.csv/json        # Ragas faithfulness subset
stage6_hit_detail.csv/json          # Per-question Hit@k detail
stage6_hit_summary.csv/json         # Aggregate Page Hit@k
stage6_metrics_summary.json         # Final metric summary
```

## Programmatic Usage

```python
from financebench_rag.config import load_config
from financebench_rag.pipeline import execute_full_pipeline

config = load_config()
outputs = execute_full_pipeline(config)

print(outputs["metrics"])
```

To call the RAG pipeline manually:

```python
from financebench_rag.config import load_config
from financebench_rag.rag_pipeline import RAGPipeline
from financebench_rag.vectorstore import build_or_load_vectorstore

config = load_config()
vectorstore = build_or_load_vectorstore([], config)
rag = RAGPipeline(config=config, vectorstore=vectorstore)

result = rag.answer_with_rag("What was the company's revenue for the reported period?", k=4)
print(result["answer"])
print(result["retrieved_chunks"])
```

## Notes And Caveats

- Evidence page numbers are standardized as zero-indexed `page_number` values to match FinanceBench evidence metadata.
- Ragas faithfulness can be slow and call-intensive, so the current pipeline evaluates only the first 20 sorted examples.
- FAISS loading uses LangChain's local deserialization path; only load indexes created by this project or from trusted sources.
- The pipeline depends on external services for dataset loading, PDF download, and LLM calls.
