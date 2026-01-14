import polars as pl

from ecd.train.trainer import _normalize_log_rows_for_polars


def test_normalize_log_rows_mixed_types_creates_dataframe() -> None:
    log_rows = [
        {"step": 1, "eval_status": "ok", "loss_total": 1.0},
        {"step": 2, "eval_status": 0, "loss_total": None},
        {"step": 3, "loss_total": 0.5},
    ]

    normalized_rows, diagnostics, schema_overrides = _normalize_log_rows_for_polars(
        log_rows
    )
    assert diagnostics.mixed_type_columns["eval_status"]

    df = pl.from_dicts(
        normalized_rows,
        schema_overrides=schema_overrides or None,
        strict=False,
        infer_schema_length=None,
    )
    assert df.height == 3
