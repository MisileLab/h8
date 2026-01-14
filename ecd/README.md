# embedding-compression-db

Neighbor-preserving embedding compression experiments for fast vector search. The goal is to learn a low-dimensional student embedding that preserves the teacher kNN structure, then evaluate end-to-end search quality and index fidelity with reproducible sweeps and plots.

## Interpreting metrics

- `vs_teacher` measures representation quality against teacher kNN truth. This is the core metric for neighbor preservation.
- `vs_student` measures index fidelity: ANN results vs brute-force in the same representation space. Use this to validate the backend configuration.

Every metrics table includes a `scope` column to prevent mixing these two interpretations.

## Quickstart (3 commands)

1. Prepare data and cache teacher top-k

```
python scripts/prepare_data.py --run-id demo
```

2. Train a student projector

```
python scripts/train_student.py --run-id demo --config configs/train/default.yaml
```

3. Run sweeps and plots

```
python scripts/run_sweep.py --run-id demo
python -c "from ecd.plots.make_plots import make_plots; make_plots('results/demo')"
```

## Debug checklist

- Run `scripts/debug_sanity.py` to confirm backend fidelity (`vs_student`).
- Verify `rep_dim` and `ckpt_path` in sweep outputs.
- Use `configs/sweep/default.yaml` to toggle strict vs loose constraint selection.

## Improving vs_teacher recall

- Increase `train.steps` in `configs/train/default.yaml` and keep hard negatives enabled.
- Tune `train.hard_negative.mix_random_ratio` and `train.hard_negative.tail_from` or `tail_to`.
- Start with one-stage evaluation before enabling two-stage rerank.

## Hard negative sweep

```
python scripts/sweep_hardneg.py --config configs/sweep/hardneg.yaml
```

## Quick hyper sweep

- `python -m scripts.sweep_quick_hyper --config configs/sweep/quick_hyper.yaml --run_id quick_hyper_01`
- Outputs:
  - `results/<run_id>/quick_hyper/summary.parquet`
  - `results/<run_id>/quick_hyper/summary.csv`
  - `results/<run_id>/quick_hyper/best.json` (best config under constraint)
- `vs_teacher` measures representation quality vs teacher neighbors
- `vs_student` measures index fidelity in the same space
- `results/<run_id>/quick_hyper/runs/<combo_id>/` stores per-combo logs
- `results/<run_id>/quick_hyper/figures/pareto_vs_teacher.png` when latency is enabled

### Track A knobs (training efficiency / longer training)

You can sweep longer training + scheduling via `configs/sweep/quick_hyper.yaml`.

- Constraint selection: `sweep.target_vs_teacher_recall10` (default 0.70)
- Additional supported grid axes:
  - `grid.steps`
  - `grid.lr`
  - `grid.lr_schedule` (constant | linear | cosine)
  - `grid.warmup_steps`
  - `grid.amp` (none | fp16 | bf16)
  - Note: `bf16` AMP is not supported on Apple `mps` in this repo; use `amp=none` (or `fp16` if supported) on MPS.
  - `grid.batch_size`

Example run (20k steps + cosine + warmup). Note the quoting on `--override ...=[...]` is required in zsh:

```bash
python -m scripts.sweep_quick_hyper \
  --config configs/sweep/quick_hyper.yaml \
  --run_id track_a_demo \
  --override 'sweep.grid.steps=[20000]' \
  --override 'sweep.grid.lr=[0.001]' \
  --override 'sweep.grid.lr_schedule=["cosine"]' \
  --override 'sweep.grid.warmup_steps=[1000]' \
  --override 'sweep.grid.amp=["bf16"]'
```

If you're on Apple MPS, use `amp=["none"]` instead (bf16 AMP will fail fast).

## GPU acceleration

- Set `device: "auto"` for CUDA or MPS when available.
- Use `device: "cuda"` or `device: "mps"` to force a backend.

## Multi-positive InfoNCE

- Samples multiple positives from teacher top-10 to strengthen neighborhood signals.
- Toggle with `train.rank.multi_positive.enabled` and `train.rank.multi_positive.num_positives`.

## Collapse diagnostics

- Check `results/<run_id>/train/figures/norm_stats.png` and `cosine_stats.png`.
- If `collapse_suspect` appears in `train_logs.parquet`, adjust `loss.rank_temperature`, `train.hard_negative.mix_random_ratio`, or `train.batch_size`.

## Scripts

- `scripts/prepare_data.py`: load HF dataset, cache teacher embeddings and top-k.
- `scripts/train_student.py`: train student projector and save checkpoints.
- `scripts/run_sweep.py`: sweep representation modes and dimensions.
- `scripts/debug_sanity.py`: backend vs brute-force checks, collapse statistics.
- `scripts/run_end_to_end.py`: end-to-end training and metrics summary.

## Results layout

```
results/<run_id>/<representation_mode>/metrics.parquet
results/<run_id>/<representation_mode>/figures/vs_teacher/pareto_vs_teacher.png
results/<run_id>/<representation_mode>/figures/vs_teacher/recall_vs_teacher_dim.png
results/<run_id>/<representation_mode>/figures/vs_teacher/overlap_vs_teacher_dim.png
results/<run_id>/<representation_mode>/figures/vs_teacher/ndcg_vs_teacher_dim.png
results/<run_id>/<representation_mode>/figures/vs_student/index_fidelity_vs_student.png
```
