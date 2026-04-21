from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pandas as pd
from openai import OpenAI

from .config import PipelineConfig
from .nebius_client import NebiusChatClient
from .utils.utils_eval import _judge_prompt, _parse_judge_response


def _normalize_ragas_score(value: Any) -> float:
    if isinstance(value, dict) and "score" in value:
        return float(value["score"])
    if hasattr(value, "score"):
        return float(getattr(value, "score"))
    return float(value)


def _run_coro_in_thread(coro: Any) -> Any:
    container: dict[str, Any] = {}

    def _runner() -> None:
        try:
            container["value"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover
            container["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in container:
        raise container["error"]
    return container.get("value")


def _build_retrieved_contexts(item: dict[str, Any]) -> list[str]:
    chunks = item.get("retrieved_chunks", [])
    contexts: list[str] = []

    if isinstance(chunks, list):
        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get("content")
                if content:
                    contexts.append(str(content))
                else:
                    contexts.append(
                        f"doc_name: {chunk.get('doc_name')}\npage_number: {chunk.get('page_number')}"
                    )
            else:
                contexts.append(str(chunk))

    return contexts or [""]


def _score_faithfulness(metric: Any, sample: dict[str, Any], ragas_llm: Any) -> float:
    try:
        from ragas.dataset_schema import SingleTurnSample

        sample_obj: Any = SingleTurnSample(**sample)
    except Exception:
        sample_obj = sample

    # Newer ragas API (sync)
    if hasattr(metric, "single_turn_score"):
        try:
            value = metric.single_turn_score(sample_obj, llm=ragas_llm)
        except TypeError:
            if hasattr(metric, "llm"):
                metric.llm = ragas_llm
            value = metric.single_turn_score(sample_obj)
        return _normalize_ragas_score(value)

    # Newer ragas API (async)
    if hasattr(metric, "single_turn_ascore"):
        try:
            coro = metric.single_turn_ascore(sample_obj, llm=ragas_llm)
        except TypeError:
            if hasattr(metric, "llm"):
                metric.llm = ragas_llm
            coro = metric.single_turn_ascore(sample_obj)
        value = _run_coro_in_thread(coro)
        return _normalize_ragas_score(value)

    # Older ragas API
    if hasattr(metric, "score"):
        value = metric.score(sample, llm=ragas_llm)
        return _normalize_ragas_score(value)

    raise AttributeError("Unsupported ragas faithfulness API: no scoring method found.")


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

    ragas_model = config.ragas_model or config.judge_model or config.generation_model
    if not ragas_model:
        subset["faithfulness_score"] = None
        subset["faithfulness_error"] = (
            "Missing RAGAS_MODEL (or JUDGE_MODEL/GENERATION_MODEL fallback) for ragas llm_factory."
        )
        return subset, None

    try:
        ragas_llm = llm_factory(
            ragas_model,
            client=OpenAI(base_url=config.api_base_url, api_key=config.api_key),
        )
    except Exception as exc:  # pragma: no cover
        subset["faithfulness_score"] = None
        subset["faithfulness_error"] = f"Failed to initialize ragas llm: {exc}"
        return subset, None

    scores: list[float | None] = []
    errors: list[str | None] = []

    for item in subset.to_dict(orient="records"):
        sample = {
            "user_input": str(item.get("question", "")),
            "response": str(item.get("rag_answer", "")),
            "retrieved_contexts": _build_retrieved_contexts(item),
        }
        try:
            score = _score_faithfulness(faithfulness, sample, ragas_llm)
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
