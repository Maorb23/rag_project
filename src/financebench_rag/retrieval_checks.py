from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_community.vectorstores import FAISS
from .utils.utils_eval import _contains_any_evidence_text



def run_retrieval_sanity_checks(
    sample_questions: pd.DataFrame,
    retriever: Any,
    k: int = 10,
    rerank: bool | None = None,
) -> pd.DataFrame:
    """
    Function to run sanity checks on the retrieval component of the RAG pipeline.
    For each question in the sample, it checks whether the expected document is retrieved,
    and whether the retrieved documents contain the expected evidence.
        Returns a DataFrame with the results of these checks for each question and retrieved document.
    """
    # If `retriever` exposes `_retrieve`, use it and return the raw retrieved lists per question.
    # By default, sanity checks should preserve the requested `k` (legacy behaviour), so
    # reranking is opt-in only for this helper.
    if hasattr(retriever, "_retrieve"):
        use_rerank = False if rerank is None else bool(rerank)
        rows: list[dict[str, Any]] = []
        for item in sample_questions.to_dict(orient="records"):
            print(f"Running sanity check for question: {item.get('question', '')}")
            question = str(item.get("question", ""))
            expected_doc = item.get("doc_name")
            expected_pages = set(item.get("evidence_page_nums", []))
            expected_evidence = item.get("evidence", "")

            retrieved = retriever._retrieve(question, k=k, rerank=use_rerank)
            for rank, doc in enumerate(retrieved, start=1):
                # _retrieve returns a list of dicts (doc_name, page_number, content).
                # Support both the new dict format and the legacy Document-like objects.
                if isinstance(doc, dict):
                    found_doc = doc.get("doc_name")
                    found_page = doc.get("page_number")
                    content = doc.get("content", "")
                else:
                    # legacy Document from langchain
                    found_doc = doc.metadata.get("doc_name")
                    found_page = doc.metadata.get("page_number")
                    content = getattr(doc, "page_content", "")

                rows.append(
                    {
                        "financebench_id": item.get("financebench_id"),
                        "question": question,
                        "rank": rank,
                        "expected_doc_name": expected_doc,
                        "retrieved_doc_name": found_doc,
                        "doc_match": str(found_doc) == str(expected_doc),
                        "expected_pages": sorted(expected_pages),
                        "retrieved_page_number": found_page,
                        "page_match": int(found_page) in expected_pages if expected_pages else False,
                        "evidence_text_approx_match": _contains_any_evidence_text(
                            content,
                            expected_evidence,
                        ),
                    }
                )
        return pd.DataFrame(rows)

    # Fallback behaviour: accept a FAISS vectorstore and run the original set of checks
    rows: list[dict[str, Any]] = []
    for item in sample_questions.to_dict(orient="records"):
        question = str(item.get("question", ""))
        expected_doc = item.get("doc_name")
        expected_pages = set(item.get("evidence_page_nums", []))
        expected_evidence = item.get("evidence", "")

        retrieved = retriever.similarity_search(question, k=k)
        for rank, doc in enumerate(retrieved, start=1):
            found_doc = doc.metadata.get("doc_name")
            found_page = doc.metadata.get("page_number")
            rows.append(
                {
                    "financebench_id": item.get("financebench_id"),
                    "question": question,
                    "rank": rank,
                    "expected_doc_name": expected_doc,
                    "retrieved_doc_name": found_doc,
                    "doc_match": str(found_doc) == str(expected_doc),
                    "expected_pages": sorted(expected_pages),
                    "retrieved_page_number": found_page,
                    "page_match": int(found_page) in expected_pages if expected_pages else False,
                    "evidence_text_approx_match": _contains_any_evidence_text(
                        doc.page_content,
                        expected_evidence,
                    ),
                }
            )

    return pd.DataFrame(rows)
