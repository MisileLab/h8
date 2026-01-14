from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from ecd.utils.io import ensure_dir


def _plot_metric(df: pl.DataFrame, metric: str, title: str, output_path: Path) -> None:
    x = df["rep_dim"].to_list()
    y = df[metric].to_list()
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel("dimension")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_pareto(df: pl.DataFrame, metric: str, title: str, output_path: Path) -> None:
    x = df["latency_p95_ms"].to_list()
    y = df[metric].to_list()
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel("latency_p95_ms")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def make_plots(run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    for mode_path in run_path.iterdir():
        if not mode_path.is_dir():
            continue
        metrics_path = mode_path / "metrics.parquet"
        if not metrics_path.exists():
            continue
        df = pl.read_parquet(metrics_path)
        if "backend_status" in df.columns:
            df = df.filter(pl.col("backend_status") == "ok")
        metric_map = {"recall@10": "recall", "overlap@10": "overlap", "ndcg@10": "ndcg"}
        for scope in ["vs_teacher", "vs_student"]:
            scope_df = df.filter(pl.col("scope") == scope)
            if scope_df.is_empty():
                continue
            scope_dir = ensure_dir(mode_path / "figures" / scope)
            for metric_key, metric_label in metric_map.items():
                title = f"{mode_path.name} | {scope} | {scope_df[0, 'backend']} | {scope_df[0, 'metric']} | {metric_key}"
                if scope == "vs_teacher" and metric_key == "recall@10":
                    _plot_pareto(
                        scope_df, metric_key, title, scope_dir / "pareto_vs_teacher.png"
                    )
                if scope == "vs_teacher" and mode_path.name in [
                    "random_projection",
                    "pca",
                ]:
                    _plot_metric(
                        scope_df.sort("rep_dim"),
                        metric_key,
                        title,
                        scope_dir / f"{metric_label}_vs_teacher_dim.png",
                    )
                if scope == "vs_student" and metric_key == "recall@10":
                    _plot_pareto(
                        scope_df,
                        metric_key,
                        title,
                        scope_dir / "index_fidelity_vs_student.png",
                    )
