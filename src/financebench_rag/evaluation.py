from __future__ import annotations

import json
from typing import Any

import pandas as pd
from openai import AsyncOpenAI

from .config import PipelineConfig
from .nebius_client import NebiusChatClient


def _judge_prompt(question: str, predicted: str, ground_truth: str) -> str:
    return (
        "You are an evaluator. Compare model answer to ground truth.\n"
        "Return JSON with keys: verdict (correct|incorrect), justification (one sentence).\n\n"
        f"Question: {question}\n\n"
        f"Model answer: {predicted}\n\n"
        f"Ground truth: {ground_truth}"
    )


def _parse_judge_response(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    try:
        payload = json.loads(text)
        verdict = str(payload.get("verdict", "incorrect")).strip().lower()
        if verdict not in {"correct", "incorrect"}:
            verdict = "incorrect"
        justification = str(payload.get("justification", "No justification provided.")).strip()
        return verdict, justification
    except Exception:
        lowered = text.lower()
        verdict = "correct" if "correct" in lowered and "incorrect" not in lowered else "incorrect"
        return verdict, text[:300] if text else "Could not parse structured output."


def compute_correctness_judgements(
    qa_df: pd.DataFrame,
    config: PipelineConfig,
    client: NebiusChatClient | None = None,
) -> tuple[pd.DataFrame, float]:
    client = client or NebiusChatClient(config)
    rows: list[dict[str, Any]] = []

    for item in qa_df.to_dict(orient="records"):
        judge_raw = client.chat(
            model=config.judge_model,
            messages=[
                {"role": "system", "content": "You are a strict grading assistant."},
                {
                    "role": "user",
                    "content": _judge_prompt(
                        question=str(item.get("question", "")),
                        predicted=str(item.get("rag_answer", "")),
                        ground_truth=str(item.get("ground_truth", "")),
                    ),
                },
            ],
            temperature=0.0,
        )
        verdict, justification = _parse_judge_response(judge_raw)
        rows.append(
            {
                "financebench_id": item.get("financebench_id"),
                "question": item.get("question"),
                "rag_answer": item.get("rag_answer"),
                "ground_truth": item.get("ground_truth"),
                "judge_verdict": verdict,
                "judge_justification": justification,
            }
        )

    result_df = pd.DataFrame(rows)
    accuracy = float((result_df["judge_verdict"] == "correct").mean()) if not result_df.empty else 0.0
    return result_df, accuracy


def compute_faithfulness_first_20(
    rag_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, float | None]:
    subset = rag_df.sort_values("financebench_id", kind="stable").head(20).copy()

    if subset.empty:
        return pd.DataFrame(), None

    try:
        from ragas.llms import llm_factory
        from ragas.metrics import faithfulness
    except Exception as exc:  # pragma: no cover
        subset["faithfulness_score"] = None
        subset["faithfulness_error"] = f"Missing ragas dependency: {exc}"
        return subset, None

    ragas_llm = llm_factory(
        AsyncOpenAI(base_url=config.api_base_url, api_key=config.api_key)
    )

    scores: list[float | None] = []
    errors: list[str | None] = []

    for item in subset.to_dict(orient="records"):
        sample = {
            "user_input": str(item.get("question", "")),
            "response": str(item.get("rag_answer", "")),
            "retrieved_contexts": [
                json.dumps(item.get("retrieved_chunks", []), ensure_ascii=False)
            ],
        }
        try:
            value = faithfulness.score(sample, llm=ragas_llm)
            if isinstance(value, dict) and "score" in value:
                score = float(value["score"])
            else:
                score = float(value)
            scores.append(score)
            errors.append(None)
        except Exception as exc:  # pragma: no cover
            scores.append(None)
            errors.append(str(exc))

    subset["faithfulness_score"] = scores
    subset["faithfulness_error"] = errors

    valid_scores = [s for s in scores if s is not None]
    mean_score = float(sum(valid_scores) / len(valid_scores)) if valid_scores else None
    return subset, mean_score


def compute_page_hit_rate(
    questions_df: pd.DataFrame,
    rag_pipeline: Any,
    k_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []

    for item in questions_df.to_dict(orient="records"):
        question = str(item.get("question", ""))
        gt_pages = set(item.get("evidence_page_nums", []))

        for k in k_values:
            retrieved = rag_pipeline.answer_with_rag(question, k=k)["retrieved_chunks"]
            ret_pages = {
                int(chunk["page_number"])
                for chunk in retrieved
                if chunk.get("page_number") is not None
            }
            hit = int(bool(gt_pages.intersection(ret_pages))) if gt_pages else 0
            detail_rows.append(
                {
                    "financebench_id": item.get("financebench_id"),
                    "k": k,
                    "hit": hit,
                    "ground_truth_pages": sorted(gt_pages),
                    "retrieved_pages": sorted(ret_pages),
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = (
        detail_df.groupby("k", as_index=False)["hit"]
        .mean()
        .rename(columns={"hit": "page_hit_rate"})
    )
    return detail_df, summary_df
