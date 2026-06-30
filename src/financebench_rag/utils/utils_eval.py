"""
File for utility functions related to evaluation of model predictions against ground truth answers in the FinanceBench RAG project. This includes functions for generating prompts for human or model-based evaluation, as well as parsing the responses from such evaluations to extract verdicts and justifications.
"""
from __future__ import annotations

import json
from typing import Any
from collections.abc import Iterable
import numpy as np

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
    """Return True if any evidence text appears (approximately) in the chunk.

    Accepts evidence as a string, list/iterable of strings, numpy object array, or pandas Series.
    Performs a case-insensitive substring check using the first 80 characters of each evidence item.
    """
    if not chunk_text:
        return False

    small_chunk = str(chunk_text).lower()

    # Normalize evidence containers (list, tuple, numpy arrays, pandas series)
    if evidence is None:
        return False

    # If it's a numpy array or pandas Series, convert to list
    try:
        if isinstance(evidence, np.ndarray):
            evidence_iter: Iterable = evidence.tolist()
        else:
            evidence_iter = evidence
    except Exception:
        evidence_iter = evidence

    # Single string
    if isinstance(evidence_iter, str):
        token = evidence_iter.strip().lower()[:80]
        return bool(token and token in small_chunk)

    # Iterable of potential evidence strings
    if isinstance(evidence_iter, Iterable):
        for item in evidence_iter:
            if not isinstance(item, str):
                continue
            token = item.strip().lower()[:80]
            if token and token in small_chunk:
                return True
        return False

    # Fallback: convert to string
    token = str(evidence).strip().lower()[:80]
    return bool(token and token in small_chunk)