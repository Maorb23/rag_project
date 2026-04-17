"""

FinanceBench dataset utilities. We first load the dataset into a DataFrame, 
then apply various transformations to normalize evidence page numbers and 
repair document links. The resulting DataFrame is used for RAG retrieval and 
evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

RAW_PDF_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
VALID_QUESTION_TYPES = {"domain-relevant", "novel-generated"}


def load_financebench_dataframe(dataset_id: str, split: str = "train") -> pd.DataFrame:
    """
    Load the FinanceBench dataset into a pandas DataFrame.
    """
    dataset = load_dataset(dataset_id, split=split)
    df = dataset.to_pandas()
    if "financebench_id" in df.columns:
        df = df.sort_values("financebench_id", kind="stable").reset_index(drop=True)
    return df


def filter_financebench_questions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the FinanceBench and retain only rows with valid question types. Sort by financebench_id if present for consistent ordering.
    """
    if "question_type" not in df.columns:
        raise ValueError("Expected question_type column in FinanceBench dataset.")
    filtered = df[df["question_type"].isin(VALID_QUESTION_TYPES)].copy()
    if "financebench_id" in filtered.columns:
        filtered = filtered.sort_values("financebench_id", kind="stable").reset_index(drop=True) # Sort by financebench_id if present for consistent ordering.
    return filtered


def _extract_evidence_pages_recursive(node: Any) -> list[int]:
    """Extract evidence page numbers from FinanceBench evidence payload.

    Expected input is a list of evidence dicts, but this function also supports
    numpy object arrays by converting them via .tolist().
    """
    if node is None: # Handle None values gracefully to avoid errors during page number extraction.
        return []

    if hasattr(node, "tolist"): # Handle numpy object arrays by converting them to lists for easier processing.
        node = node.tolist()

    if isinstance(node, dict): # If the node is a dict, check if it has the 'evidence_page_num' key and extract it. Otherwise, continue searching recursively in its values.
        node = [node] # Wrap dict in a list to process it uniformly with lists of dicts.

    if not isinstance(node, list): # If the node is not a list at this point, it means it's an unexpected type (e.g., a string or number), so we return an empty list since we can't extract page numbers from it.
        return []

    pages: list[int] = []

    for item in node:
        if not isinstance(item, dict): # If the item is not a dict, we skip it since we expect evidence items to be dicts containing metadata.
            continue

        value = item.get("evidence_page_num")
        if value is None:
            continue

        values = value if isinstance(value, list) else [value]
        for candidate in values:
            pages.append(int(candidate))


    return pages


def normalize_evidence_pages(df: pd.DataFrame, evidence_col: str = "evidence") -> pd.DataFrame:
    df = df.copy()
    if evidence_col not in df.columns:
        df["evidence_page_nums"] = [[] for _ in range(len(df))]
        return df

    normalized_pages: list[list[int]] = []
    for raw in df[evidence_col].tolist():
        pages = _extract_evidence_pages_recursive(raw)
        pages = sorted(set(p for p in pages if isinstance(p, int) and p >= 0))
        normalized_pages.append(pages)

    df["evidence_page_nums"] = normalized_pages
    return df


def _is_dead_doc_link(link: Any) -> bool:
    if not isinstance(link, str) or not link.strip():
        return True
    normalized = link.strip().lower()
    return not normalized.startswith("http") or "404" in normalized


def _to_pdf_filename(doc_name: str) -> str:
    doc_name = str(doc_name).strip()
    if doc_name.lower().endswith(".pdf"):
        return doc_name
    return f"{doc_name}.pdf"


def repair_doc_links(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "doc_name" not in df.columns:
        raise ValueError("Expected doc_name column in FinanceBench dataset.")

    df = df.copy()
    if "doc_link" not in df.columns:
        df["doc_link"] = ""

    required_doc_names = sorted({str(name).strip() for name in df["doc_name"].dropna().unique().tolist()})
    mapping_rows: list[dict[str, Any]] = []

    doc_name_to_link: dict[str, str] = {}
    for doc_name in required_doc_names:
        pdf_name = _to_pdf_filename(doc_name)
        repaired_url = f"{RAW_PDF_BASE}/{pdf_name}"
        doc_name_to_link[doc_name] = repaired_url
        mapping_rows.append(
            {
                "doc_name": doc_name,
                "pdf_name": pdf_name,
                "repaired_doc_link": repaired_url,
            }
        )

    repaired_links: list[str] = []
    was_repaired: list[bool] = []
    for doc_name, current_link in zip(df["doc_name"].tolist(), df["doc_link"].tolist()):
        doc_name = str(doc_name).strip()
        if _is_dead_doc_link(current_link):
            repaired_links.append(doc_name_to_link[doc_name])
            was_repaired.append(True)
        else:
            repaired_links.append(str(current_link).strip())
            was_repaired.append(False)

    df["doc_link"] = repaired_links
    df["doc_link_repaired"] = was_repaired

    mapping_df = pd.DataFrame(mapping_rows)
    return df, mapping_df


def build_doc_metadata_table(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["doc_name", "company", "doc_period"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required document columns: {missing}")

    table = (
        df[required_cols]
        .dropna(subset=["doc_name"])
        .drop_duplicates(subset=["doc_name"], keep="first")
        .sort_values("doc_name", kind="stable")
        .reset_index(drop=True)
    )
    return table


def prepare_stage1_dataset(
    dataset_id: str,
    split: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = load_financebench_dataframe(dataset_id=dataset_id, split=split)
    filtered_df = filter_financebench_questions(full_df)
    filtered_df = normalize_evidence_pages(filtered_df)
    filtered_df, mapping_df = repair_doc_links(filtered_df)
    return filtered_df, mapping_df
