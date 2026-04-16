from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _json_default(value: Any) -> Any:
    """Convert common non-JSON-native objects to serializable values."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
    except Exception:
        pass

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    return str(value)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def save_dataframe_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_dataframe_json(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = df.to_dict(orient="records")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=_json_default)
