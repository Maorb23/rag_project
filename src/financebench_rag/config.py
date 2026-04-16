from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


@dataclass(slots=True)
class PipelineConfig:
    dataset_id: str = "PatronusAI/financebench"
    dataset_split: str = "train"
    api_base_url: str = ""
    api_key: str = ""
    generation_model: str = ""
    judge_model: str = ""
    ragas_model: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_default_k: int = 4
    retrieval_hit_k_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    data_dir: Path = Path("data")
    pdf_dir: Path = Path("data/pdfs")
    vectorstore_dir: Path = Path("vectorstore")
    results_dir: Path = Path("results")

    def validate_required_secrets(self) -> None:
        if not self.api_base_url or not self.api_key:
            raise ValueError(
                "Missing NEBIUS_BASE_URL or NEBIUS_API_KEY. Configure environment before model calls."
            )


def _parse_k_values(raw: str | None) -> list[int]:
    if not raw:
        return [1, 3, 5]
    values: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values or [1, 3, 5]


def load_config(env_file: str | None = ".env") -> PipelineConfig:
    if load_dotenv and env_file:
        load_dotenv(env_file)

    return PipelineConfig(
        dataset_id=os.getenv("FINANCEBENCH_DATASET_ID", "PatronusAI/financebench"),
        dataset_split=os.getenv("FINANCEBENCH_DATASET_SPLIT", "train"),
        api_base_url=os.getenv("NEBIUS_BASE_URL", "").rstrip("/"),
        api_key=os.getenv("NEBIUS_API_KEY", ""),
        generation_model=os.getenv("GENERATION_MODEL", ""),
        judge_model=os.getenv("JUDGE_MODEL", ""),
        ragas_model=os.getenv("RAGAS_MODEL", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        retrieval_default_k=int(os.getenv("RETRIEVAL_DEFAULT_K", "4")),
        retrieval_hit_k_values=_parse_k_values(os.getenv("RETRIEVAL_HIT_K_VALUES")),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        pdf_dir=Path(os.getenv("PDF_DIR", "data/pdfs")),
        vectorstore_dir=Path(os.getenv("VECTORSTORE_DIR", "vectorstore")),
        results_dir=Path(os.getenv("RESULTS_DIR", "results")),
    )
