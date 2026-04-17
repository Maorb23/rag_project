"""
Build a side-by-side comparison table of questions and their answers from different models.
"""

from __future__ import annotations

import pandas as pd


def build_side_by_side_table(
    questions_df: pd.DataFrame,
    naive_df: pd.DataFrame,
    rag_df: pd.DataFrame,
) -> pd.DataFrame:
    left = questions_df[["financebench_id", "question", "answer"]].rename(
        columns={"answer": "ground_truth"}
    )

    merged = left.merge(
        naive_df[["financebench_id", "naive_answer"]],
        on="financebench_id",
        how="left",
    ).merge(
        rag_df[["financebench_id", "rag_answer"]],
        on="financebench_id",
        how="left",
    )

    return merged[["financebench_id", "question", "ground_truth", "naive_answer", "rag_answer"]]
