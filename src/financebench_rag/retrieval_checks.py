from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_community.vectorstores import FAISS
from .utils.utils_eval import _contains_any_evidence_text



def run_retrieval_sanity_checks(
    sample_questions: pd.DataFrame,
    vectorstore: FAISS,
    k: int = 4,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in sample_questions.to_dict(orient="records"):
        question = str(item.get("question", ""))
        expected_doc = item.get("doc_name")
        expected_pages = set(item.get("evidence_page_nums", []))
        expected_evidence = item.get("evidence", "")

        retrieved = vectorstore.similarity_search(question, k=k)
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
