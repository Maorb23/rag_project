"""
file for dataset-related utility functions, such as evidence page number extraction and doc link repair.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

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

def _normalize_pdf_name(doc_name: str) -> str:
    name = str(doc_name).strip()
    if name.lower().endswith(".pdf"):
        return name
    return f"{name}.pdf"