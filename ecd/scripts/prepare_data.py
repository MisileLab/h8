from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ecd.data.dataset import load_embeddings_dataset
from ecd.embed.cache import load_or_prepare_teacher_cache, write_meta
from ecd.utils.config import load_config
from ecd.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--dataset-config", type=str, default="configs/dataset/default.yaml"
    )
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_overrides = list(args.override) + [f"run_id={run_id}"]
    config = load_config([args.dataset_config], dataset_overrides)
    cache_dir = config["cache_dir"].format(run_id=run_id)
    bundle = load_embeddings_dataset(
        name=config["name"],
        split=config["split"],
        embedding_fields_priority=config["embedding_fields_priority"],
        text_fields=config["text_fields"],
        max_rows=config["max_rows"],
    )
    embeddings, topk = load_or_prepare_teacher_cache(
        bundle, cache_dir=cache_dir, k=config["teacher_topk"], metric=config["metric"]
    )
    write_meta(cache_dir, bundle)
    save_json(
        Path(cache_dir) / "summary.json",
        {
            "run_id": run_id,
            "embeddings_shape": list(embeddings.shape),
            "topk_shape": list(topk.shape),
        },
    )


if __name__ == "__main__":
    main()
