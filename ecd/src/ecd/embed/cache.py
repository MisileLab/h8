from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ecd.data.dataset import DatasetBundle
from ecd.utils.device import estimate_batch_size, resolve_device
from ecd.utils.io import ensure_dir, load_numpy, save_numpy, save_parquet


def compute_teacher_topk(
    embeddings: np.ndarray,
    k: int,
    metric: str = "cosine",
    batch_size: Optional[int] = None,
    device: str = "auto",
) -> np.ndarray:
    resolved_device = resolve_device(device)
    matrix = torch.tensor(embeddings, dtype=torch.float32, device=resolved_device)

    if metric == "cosine":
        matrix = F.normalize(matrix, p=2, dim=-1)

    n, dim = matrix.shape

    if batch_size is None:
        batch_size = estimate_batch_size(
            num_items=n,
            embedding_dim=dim,
            device=resolved_device,
            dtype=torch.float32,
        )

    topk_indices = torch.zeros((n, k), dtype=torch.int32, device="cpu")

    for batch_start in tqdm(
        range(0, n, batch_size), desc=f"teacher topk ({resolved_device.type})"
    ):
        batch_end = min(batch_start + batch_size, n)
        batch = matrix[batch_start:batch_end]

        scores = torch.matmul(batch, matrix.T)

        batch_indices = torch.arange(batch_start, batch_end, device=resolved_device)
        scores[torch.arange(batch.shape[0], device=resolved_device), batch_indices] = (
            float("-inf")
        )

        _, indices = torch.topk(scores, k, dim=1, largest=True, sorted=True)

        topk_indices[batch_start:batch_end] = indices.cpu().to(torch.int32)

    return topk_indices.numpy()


def load_or_prepare_teacher_cache(
    bundle: DatasetBundle,
    cache_dir: str | Path,
    k: int,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    cache_path = Path(cache_dir)
    ensure_dir(cache_path)
    embedding_path = cache_path / "teacher_embeddings.npy"
    topk_path = cache_path / "teacher_topk.npy"
    if embedding_path.exists():
        embeddings = load_numpy(embedding_path)
    else:
        embeddings = bundle.embeddings.astype(np.float32)
        save_numpy(embedding_path, embeddings)
    if topk_path.exists():
        topk = load_numpy(topk_path)
    else:
        topk = compute_teacher_topk(embeddings, k=k, metric=metric)
        save_numpy(topk_path, topk.astype(np.int32))
    return embeddings, topk.astype(np.int32)


def write_meta(cache_dir: str | Path, bundle: DatasetBundle) -> None:
    cache_path = Path(cache_dir)
    ensure_dir(cache_path)
    data = pl.DataFrame(
        {
            "doc_id": bundle.doc_ids,
            "text": bundle.texts,
            "text_len": [len(text) for text in bundle.texts],
        }
    )
    save_parquet(cache_path / "meta.parquet", data)
