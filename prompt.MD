# FinanceBench RAG System — Full Pipeline

## Role

You are a senior data scientist with 5 years of experience specializing in Large Language Models and Retrieval-Augmented Generation (RAG).

## Objective

Build a fully self-contained Python project that answers questions from the FinanceBench dataset, both with and without RAG. Deliver all pipeline stages as .py modules and execute the full pipeline in pipeline_notebook.ipynb.

## Pipeline Stages

### Stage 1 — Dataset Preparation

Load the FinanceBench dataset from HuggingFace.
Drop all rows where question_type == "metrics-generated". Keep only domain-relevant and novel-generated questions.
Fix dead doc_link URLs: replace any broken links with the corresponding file path from:
https://github.com/patronus-ai/financebench/tree/main/pdfs

⚠️ That repo contains more documents than the dataset references — only use PDFs whose doc_name appears in your filtered dataset.




### Stage 2 — Naive Generation (No Retrieval)

Model: Llama-3.3-70B-Instruct via Nebius Token Factory

Sort questions by financebench_id and select the first 5 of each remaining type:

5 × domain-relevant
5 × novel-generated


Feed each question directly into the model — no context, no retrieval.
Record the model's output alongside the ground truth (answer field).


📝 The model may refuse to answer, hedge, or ask for more information. This is a valid and recordable result.


### Stage 3 — Document Embedding & Vector Store

Embedding model: BAAI/bge-small-en-v1.5 (HuggingFace)
Vector store: LangChain FAISS
3.1 — Load PDFs

Use PyPDFLoader, producing one Document per page.
Only load PDFs whose doc_name appears in your filtered dataset.

3.2 — Attach Metadata
Before splitting, attach the following metadata fields to each page:
FieldSourcedoc_nameDataset columncompanyDataset columndoc_periodDataset columnpage_number0-indexed (to match evidence_page_num)

⚠️ PyPDFLoader attaches a page field by default — standardize on page_number for consistency.

3.3 — Chunk

Splitter: RecursiveCharacterTextSplitter
chunk_size = 1000, chunk_overlap = 150
Chunks inherit all page-level metadata.

3.4 — Store & Persist

Build a FAISS vector store from the chunks using the embedding model.
Save with vectorstore.save_local(...) to avoid re-embedding on subsequent runs.

3.5 — Retrieval Sanity Check
Pick 2–3 questions from the dataset and retrieve the top-k chunks for each. Verify:

✅ Do retrieved chunks come from the correct document (matching doc_name)?
✅ Do they contain or approximate the evidence text from the dataset?
✅ Do they come from the correct page (chunk.page_number vs. evidence_page_num)?


### Stage 4 — RAG Pipeline

Generation model: Llama-3.3-70B-Instruct via Nebius Token Factory
Embedding model: BAAI/bge-small-en-v1.5 (same as Stage 3)
4.1 — Core Function
Implement the following function:
pythondef answer_with_rag(query: str, k: int = 4) -> dict:
    """
    Returns:
        {
            "answer": str,                  # model's final answer
            "retrieved_chunks": list[dict]  # each with doc_name and page_number
        }
    """
4.2 — Prompt Construction

Separate retrieved chunks clearly (e.g., with --- dividers).
Include doc_name metadata per chunk so the model knows which filing it came from.
If retrieval returns no chunks, explicitly tell the model there is no available context — do not pass an empty block silently.

4.3 — System Prompt Requirements
Instruct the model to:

Answer only from the provided context.
Explicitly state when the context doesn't contain the answer, rather than guessing.
Keep answers concise and cite the source document for each fact.


### Stage 5 — Side-by-Side Comparison

Run the same 10 questions from Stage 2 through the RAG pipeline.
Present results in a structured table:

QuestionGround TruthNaive AnswerRAG Answer............

### Stage 6 — Evaluation

Run all questions in the filtered dataset through the RAG pipeline, then evaluate using the three measures below.

Measure 1 — Correctness (LLM-as-Judge)
Judge model: DeepSeek-V3-0324 via Nebius Token Factory

Write a judge prompt that receives the question, the model's answer, and the ground truth.
Return a binary verdict: correct or incorrect, plus a one-sentence justification.


Measure 2 — Faithfulness (Ragas)
Scope: First 20 examples only (sorted by financebench_id), due to API call volume.

Use ragas.faithfulness from the Ragas collections API.
Call .score() — the synchronous method, not .ascore().
Configure the Ragas LLM as follows:

pythonfrom ragas.llms import llm_factory
from openai import AsyncOpenAI

"""
Use DeepSeek-V3-0324 wrapped via AsyncOpenAI pointed at Nebius Token Factory
"""
ragas_llm = llm_factory(AsyncOpenAI(base_url="<nebius_url>", api_key="<key>"))

⚠️ Ragas Faithfulness makes multiple LLM calls per example — it is slow. Limit to 20 examples.


Measure 3 — Retrieval Hit-Rate (Page Hit@k)
Ground truth: evidence_page_num field inside the evidence column.

For each question, retrieve top-k chunks and check if any chunk's page_number matches any evidence_page_num for that question.
Compute Page Hit@k:

1 if any retrieved chunk is from a correct evidence page
0 otherwise


Average across the full dataset.
Report results for multiple values of k: e.g., k = 1, 3, 5.


📝 Some questions have multiple evidence items across different pages — a single match on any of them counts as a hit.


## Deliverables

- pipeline_notebook.ipynb-  Full end-to-end execution of all stages*
- .py modules- Modular, importable pipeline 
- vectorstore/- Saved FAISS index