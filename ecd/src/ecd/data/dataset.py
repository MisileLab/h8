from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class DatasetBundle:
    doc_ids: np.ndarray
    texts: List[str]
    embeddings: np.ndarray


def _pick_embedding_field(columns: Sequence[str], priority: Sequence[str]) -> str:
    for name in priority:
        if name in columns:
            return name
    raise ValueError(f"No embedding column found. Available: {columns}")


def _parse_embedding(value: object) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def load_embeddings_dataset(
    name: str,
    split: str,
    embedding_fields_priority: Sequence[str],
    text_fields: Sequence[str],
    max_rows: Optional[int] = None,
) -> DatasetBundle:
    dataset = load_dataset(name, split=split)
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, len(dataset))))
    embedding_field = _pick_embedding_field(
        dataset.column_names, embedding_fields_priority
    )
    texts: List[str] = []
    embeddings: List[np.ndarray] = []
    doc_ids: List[int] = []
    for idx in tqdm(range(len(dataset)), desc="load dataset"):
        row: Dict[str, object] = dataset[int(idx)]
        doc_ids.append(idx)
        text_parts = [str(row.get(field, "")) for field in text_fields]
        texts.append("\n".join(text_parts).strip())
        embeddings.append(_parse_embedding(row[embedding_field]))
    embeddings_array = np.stack(embeddings).astype(np.float32)
    return DatasetBundle(
        doc_ids=np.asarray(doc_ids, dtype=np.int64),
        texts=texts,
        embeddings=embeddings_array,
    )


def iter_batches(array: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, array.shape[0], batch_size):
        yield array[start : start + batch_size]
