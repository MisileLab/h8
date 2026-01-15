#!/usr/bin/env python3
"""
Extract metrics from Track-B training logs for run_track_b.nu
Usage: python scripts/extract_trackb_metrics.py <log_dir>
"""

import sys
import json
from pathlib import Path

try:
    import polars as pl
except ImportError:
    print("ERROR: polars not installed. Install with: pip install polars")
    sys.exit(1)


def extract_from_parquet(parquet_path: Path):
    try:
        df = pl.read_parquet(parquet_path)
    except Exception as e:
        return {"status": "read_error", "error": str(e)}

    if df.is_empty():
        return {"status": "empty_logs"}

    try:
        recall_col = df.get_column("vs_teacher_recall@10")
        recall_values = recall_col.drop_nulls()
    except Exception:
        return {
            "status": "no_eval",
            "vs_teacher_recall10": None,
            "vs_teacher_ndcg10": None,
        }

    try:
        ndcg_col = df.get_column("vs_teacher_ndcg@10")
        ndcg_values = ndcg_col.drop_nulls()
    except Exception:
        ndcg_values = None

    if recall_values.len() == 0:
        return {
            "status": "no_eval",
            "vs_teacher_recall10": None,
            "vs_teacher_ndcg10": None,
        }

    best_recall = recall_values.max()
    final_recall = recall_values.last()
    final_ndcg = ndcg_values.last() if ndcg_values is not None else None

    return {
        "status": "success",
        "vs_teacher_recall10": float(best_recall),
        "vs_teacher_ndcg10": float(final_ndcg) if final_ndcg is not None else None,
    }


def extract_from_jsonl(jsonl_path: Path):
    """Extract metrics from train_logs.jsonl"""
    try:
        with open(jsonl_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        return {"status": "read_error", "error": str(e)}

    eval_lines = [l for l in lines if "vs_teacher_recall@10" in l]

    if not eval_lines:
        return {
            "status": "no_eval",
            "vs_teacher_recall10": None,
            "vs_teacher_ndcg10": None,
        }

    last_eval = eval_lines[-1]

    return {
        "status": "success",
        "vs_teacher_recall10": last_eval.get("vs_teacher_recall@10"),
        "vs_teacher_ndcg10": last_eval.get("vs_teacher_ndcg@10"),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_trackb_metrics.py <log_dir>")
        sys.exit(1)

    log_dir = Path(sys.argv[1])

    if not log_dir.exists():
        print(json.dumps({"status": "missing_logs"}))
        return

    parquet_path = log_dir / "train_logs.parquet"
    if parquet_path.exists():
        result = extract_from_parquet(parquet_path)
        print(json.dumps(result))
        return

    jsonl_path = log_dir / "train_logs.jsonl"
    if jsonl_path.exists():
        result = extract_from_jsonl(jsonl_path)
        print(json.dumps(result))
        return

    print(json.dumps({"status": "missing_logs"}))


if __name__ == "__main__":
    main()
