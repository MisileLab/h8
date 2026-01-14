from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ecd.data.dataset import load_embeddings_dataset
from ecd.embed.cache import load_or_prepare_teacher_cache, write_meta
from ecd.train.trainer import train_student
from ecd.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--dataset-config", type=str, default="configs/dataset/default.yaml"
    )
    parser.add_argument(
        "--model-config", type=str, default="configs/model/default.yaml"
    )
    parser.add_argument("--config", type=str, default="configs/train/default.yaml")
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_overrides = list(args.override) + [f"run_id={run_id}"]
    dataset_cfg = load_config([args.dataset_config], dataset_overrides)
    model_cfg = load_config([args.model_config])
    train_config_path = args.train_config or args.config
    train_cfg = load_config([train_config_path], args.override)
    cache_dir = dataset_cfg["cache_dir"].format(run_id=run_id)
    bundle = load_embeddings_dataset(
        name=dataset_cfg["name"],
        split=dataset_cfg["split"],
        embedding_fields_priority=dataset_cfg["embedding_fields_priority"],
        text_fields=dataset_cfg["text_fields"],
        max_rows=dataset_cfg["max_rows"],
    )
    embeddings, topk = load_or_prepare_teacher_cache(
        bundle,
        cache_dir=cache_dir,
        k=dataset_cfg["teacher_topk"],
        metric=dataset_cfg["metric"],
    )
    write_meta(cache_dir, bundle)
    if model_cfg.get("in_dim") is None:
        model_cfg["in_dim"] = embeddings.shape[1]
    output_dir = Path("results") / run_id / "checkpoints"
    log_dir = Path("results") / run_id / "train"
    train_student(
        embeddings,
        topk,
        model_cfg,
        train_cfg,
        output_dir=output_dir,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()
