from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_numpy(path: str | Path, array: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_numpy(path: str | Path) -> np.ndarray:
    return np.load(path)


def save_parquet(path: str | Path, data: pl.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(path)


def load_parquet(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)
