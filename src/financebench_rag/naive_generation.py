from __future__ import annotations

from typing import Any

import pandas as pd

from .config import PipelineConfig
from .nebius_client import NebiusChatClient


def sample_stage2_questions(df: pd.DataFrame, n_per_type: int = 5) -> pd.DataFrame:
    if "question_type" not in df.columns:
        raise ValueError("Expected question_type column.")
    if "financebench_id" in df.columns:
        df = df.sort_values("financebench_id", kind="stable")

    parts = []
    for qtype in ["domain-relevant", "novel-generated"]:
        subset = df[df["question_type"] == qtype].head(n_per_type)
        parts.append(subset)

    return pd.concat(parts, axis=0).reset_index(drop=True)


def run_naive_generation(
    questions_df: pd.DataFrame,
    config: PipelineConfig,
    client: NebiusChatClient | None = None,
) -> pd.DataFrame:
    if "question" not in questions_df.columns:
        raise ValueError("Expected question column.")

    client = client or NebiusChatClient(config)
    records: list[dict[str, Any]] = []

    for row in questions_df.to_dict(orient="records"):
        question = str(row["question"])
        answer = client.chat(
            model=config.generation_model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer concisely. If uncertain, say you do not know.",
                },
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )
        records.append(
            {
                "financebench_id": row.get("financebench_id"),
                "question_type": row.get("question_type"),
                "question": question,
                "ground_truth": row.get("answer", ""),
                "naive_answer": answer,
            }
        )

    return pd.DataFrame(records)
