from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

RAW_PDF_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
VALID_QUESTION_TYPES = {"domain-relevant", "novel-generated"}


def load_financebench_dataframe(dataset_id: str, split: str = "train") -> pd.DataFrame:
    dataset = load_dataset(dataset_id, split=split)
    df = dataset.to_pandas()
    if "financebench_id" in df.columns:
        df = df.sort_values("financebench_id", kind="stable").reset_index(drop=True)
    return df


def filter_financebench_questions(df: pd.DataFrame) -> pd.DataFrame:
    if "question_type" not in df.columns:
        raise ValueError("Expected question_type column in FinanceBench dataset.")
    filtered = df[df["question_type"].isin(VALID_QUESTION_TYPES)].copy()
    if "financebench_id" in filtered.columns:
        filtered = filtered.sort_values("financebench_id", kind="stable").reset_index(drop=True)
    return filtered


def _parse_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _extract_evidence_pages_recursive(node: Any) -> list[int]:
    pages: list[int] = []

    if isinstance(node, dict):
        for key, value in node.items():
            if key == "evidence_page_num":
                if isinstance(value, list):
                    for item in value:
                        try:
                            pages.append(int(item))
                        except Exception:
                            continue
                else:
                    try:
                        pages.append(int(value))
                    except Exception:
                        pass
            else:
                pages.extend(_extract_evidence_pages_recursive(value))

    elif isinstance(node, list):
        for item in node:
            pages.extend(_extract_evidence_pages_recursive(item))

    return pages


def normalize_evidence_pages(df: pd.DataFrame, evidence_col: str = "evidence") -> pd.DataFrame:
    if evidence_col not in df.columns:
        df = df.copy()
        df["evidence_page_nums"] = [[] for _ in range(len(df))]
        return df

    df = df.copy()
    normalized_pages: list[list[int]] = []

    for raw in df[evidence_col].tolist():
        parsed = _parse_json_like(raw)
        pages = sorted(set(_extract_evidence_pages_recursive(parsed)))
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
