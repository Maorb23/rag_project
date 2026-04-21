"""
File for utility functions related to evaluation of model predictions against ground truth answers in the FinanceBench RAG project. This includes functions for generating prompts for human or model-based evaluation, as well as parsing the responses from such evaluations to extract verdicts and justifications.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from openai import OpenAI

from ..config import PipelineConfig
from ..nebius_client import NebiusChatClient

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
    

def _contains_any_evidence_text(chunk_text: str, evidence: Any) -> bool:
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    small_chunk = chunk_text.lower()
    token = evidence.strip().lower()[:80]
    return token in small_chunk if token else False