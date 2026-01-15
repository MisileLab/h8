# Track-B v1 Experiment Runs Guide

This guide explains how to run Track-B v1 experiments using the orchestration script.

## Quick Start

Run all Track-B experiments with a single command:

```bash
bash scripts/run_track_b.sh
```

That's it! The script will:
1. Run sanity checks (if available)
2. Execute all Track-B experiments sequentially
3. Generate `summary.csv` and `summary.txt` with results

## Experiment Matrix

The script runs the following experiments in sequence:

### A) Sanity Check
- **Purpose**: Quick validation of Track-B implementation
- **Script**: `python scripts/check_trackb_sanity.py` (if exists)
- **Note**: Failure here is a warning, not a block

### B) Core Runs

#### 1. Core Run 1: 20k steps, seed 0
- `STEPS=20000`, `SEEDS=0`
- `LR=0.0005`, `LR_SCHEDULE=cosine`, `WARMUP_STEPS=2000`, `AMP=none`
- `TRACK_B_TAU=0.07`, `TRACK_B_K=50`, `TRACK_B_NEG_M=1024`, `TRACK_B_QUEUE_SIZE=32768`

#### 2. Core Run 2: 20k steps, seeds 0,1,2
- `STEPS=20000`, `SEEDS=0,1,2`
- Same hyperparameters as Core Run 1

#### 3. Core Run 3: 50k steps, seeds 0,1,2
- `STEPS=50000`, `SEEDS=0,1,2`
- Same hyperparameters as Core Run 1

### C) Minimal Tuning

#### 4. Tuning A: Tau = 0.05
- `STEPS=20000`, `SEEDS=0`
- `TRACK_B_TAU=0.05` (lower temperature)

#### 5. Tuning B: Tau = 0.10
- `STEPS=20000`, `SEEDS=0`
- `TRACK_B_TAU=0.10` (higher temperature)

## Output Files

### summary.csv
Machine-readable CSV file with all experiment results:

| Column | Description |
|--------|-------------|
| timestamp | Run completion time |
| run_id | Unique run identifier |
| experiment | Experiment name (e.g., core_20k_seed0) |
| steps | Training steps |
| seeds | Seed values used |
| lr | Learning rate |
| warmup | Warmup steps |
| tau | Track-B temperature |
| k | Track-B k (positives) |
| neg_m | Track-B m (negatives) |
| queue_size | Track-B queue size |
| false_neg_mode | False negative filter mode |
| threshold | False negative threshold |
| status | success / failed / skipped / etc. |
| vs_teacher_recall10 | Best vs_teacher recall@10 |
| vs_teacher_ndcg10 | Final vs_teacher ndcg@10 |

### summary.txt
Human-readable summary including:
- Top 3 runs by vs_teacher_recall@10
- Average metrics across successful runs
- Failed/incomplete runs
- Full run list

## Options

### Dry Run Mode
Preview what will be executed without running:

```bash
bash scripts/run_track_b.sh --dry-run
# or
bash scripts/run_track_b.sh -d
```

### Quick Mode
Short training (10 steps) for testing:

```bash
bash scripts/run_track_b.sh --quick
# or
bash scripts/run_track_b.sh -q
```

Useful for:
- Verifying script works
- Testing infrastructure
- Quick smoke tests

### Force Rerun
Rerun all experiments even if they already completed:

```bash
bash scripts/run_track_b.sh --force
# or
bash scripts/run_track_b.sh -f
```

### Help
Show all options:

```bash
bash scripts/run_track_b.sh --help
# or
bash scripts/run_track_b.sh -h
```

### Quick Mode
Short training (10 steps) for testing:

```bash
nu scripts/run_track_b.nu --quick
# or
nu scripts/run_track_b.nu -q
```

Useful for:
- Verifying the script works
- Testing infrastructure
- Quick smoke tests

### Force Rerun
Rerun all experiments even if they already completed:

```bash
nu scripts/run_track_b.nu --force
# or
nu scripts/run_track_b.nu -f
```

### Help
Show all options:

```bash
nu scripts/run_track_b.nu --help
# or
nu scripts/run_track_b.nu -h
```

## Resume / Skip Logic

By default, the script **skips** experiments that:
- Have a completed run directory
- Have `logs/train_logs.parquet` or `logs/train_logs.jsonl`
- Have successful metric extraction (`status=success`)

Experiments are **retried** if:
- Run directory exists but metrics are missing
- Metrics show failure or incomplete status
- Previous run crashed

To disable skip logic and rerun everything, use `--force`.

## Environment Variables

The script sets these environment variables (passed to `run_track_a.sh`):

```bash
TRACK_B_ENABLE=true              # Enable Track-B v1
TRACK_B_DIM=128                 # Output dimension
TRACK_B_K=50                    # Number of teacher top-k positives
TRACK_B_NEG_M=1024              # Number of queue negatives
TRACK_B_QUEUE_SIZE=32768        # Queue size
TRACK_B_TAU=0.07                # Temperature (varies per experiment)
TRACK_B_FALSE_NEG_MODE=threshold # False negative filter mode
TRACK_B_FALSE_NEG_THRESHOLD=0.8  # False negative threshold
STEPS=20000                     # Training steps (varies per experiment)
SEEDS=0                        # Seeds (varies per experiment)
LR=0.0005                      # Learning rate
LR_SCHEDULE=cosine               # LR schedule
WARMUP_STEPS=2000              # Warmup steps
AMP=none                        # Mixed precision mode
SAVE_EVERY=2000                 # Save checkpoint frequency
EVAL_EVERY=1000                # Evaluation frequency
```

## RUN_ID Format

Each run gets a unique ID:

```
trackB_v1_{experiment_name}_s{steps}_seeds{seeds}_tau{tau}_m{neg_m}_q{queue_size}_{timestamp}
```

Example:
```
trackB_v1_core_20k_seed0_s20000_seeds0_tau0.07_m1024_q32768_20260115_143000
```

## Results Location

- **Run directories**: `runs/{run_id}/`
  - `logs/train_logs.parquet` or `logs/train_logs.jsonl` - Training logs
  - `logs/train_summary.json` - Training summary
  - `checkpoints/student_best.pt` - Best checkpoint
  - `checkpoints/student_step_{N}.pt` - Periodic checkpoints

- **Summary files** (repo root):
  - `summary.csv` - Machine-readable summary
  - `summary.txt` - Human-readable summary

## Metrics Extraction

The script parses metrics from:

1. **Preferred**: `runs/{run_id}/logs/train_logs.parquet`
   - Uses Python + Polars to read parquet
   - Extracts best and final `vs_teacher_recall@10`, `vs_teacher_ndcg@10`

2. **Fallback**: `runs/{run_id}/logs/train_logs.jsonl`
   - Parses JSONL line by line
   - Finds last line with eval metrics

3. **Missing**: If neither file exists, marks as `missing_logs`

**Important**: The script requires `polars` (Python) to read parquet files. If Python or polars is unavailable, it falls back to JSONL.

## Debugging Tips

### Experiment Failed / Missing Metrics

1. **Check run directory exists**:
   ```bash
   ls -la runs/{run_id}/
   ```

2. **Check logs directory**:
   ```bash
   ls -la runs/{run_id}/logs/
   ```

3. **View training logs**:
   ```bash
   cat runs/{run_id}/logs/train_logs.jsonl | head -20
   # or
   python -c "import polars as pl; df = pl.read_parquet('runs/{run_id}/logs/train_logs.parquet'); print(df)"
   ```

4. **Check for errors**:
   ```bash
   cat runs/{run_id}/logs/sweep_seed0.log
   ```

### Metrics Not Found

If `status=missing_logs` or `status=parse_error`:
1. Verify `eval_every_steps` was set in the run
2. Check if evaluation actually ran (search for "vs_teacher" in logs)
3. Check if training completed (look for final step in logs)

### Run Crashed

If `status=failed`:
1. Check the sweep log: `runs/{run_id}/logs/sweep_seed*.log`
2. Look for Python exceptions or CUDA errors
3. Check GPU availability: `nvidia-smi`

## Manual Run of Single Experiment

If you want to manually run a single experiment:

```bash
TRACK_B_ENABLE=true \
TRACK_B_DIM=128 \
TRACK_B_K=50 \
TRACK_B_NEG_M=1024 \
TRACK_B_QUEUE_SIZE=32768 \
TRACK_B_TAU=0.07 \
TRACK_B_FALSE_NEG_MODE=threshold \
TRACK_B_FALSE_NEG_THRESHOLD=0.8 \
STEPS=20000 \
SEEDS=0 \
LR=0.0005 \
LR_SCHEDULE=cosine \
WARMUP_STEPS=2000 \
AMP=none \
RUN_ID=my_custom_run \
bash scripts/run_track_a.sh sweep
```

Note: This uses `TRACK_B_NEG_M` but the orchestration script also passes `TRACK_B_M` for compatibility with older configurations. Both work the same way.

## Troubleshooting

### Python/Polars Not Found

The metrics extraction uses Python + Polars. Install:
```bash
pip install polars
```

Or ensure your virtual environment is activated.

### GPU Issues

If training fails due to GPU:
1. Check `nvidia-smi` for CUDA availability
2. Use `AMP=none` if bf16/fp16 issues occur
3. Reduce batch size via `BATCH_SIZE` env var

## References

- [Track-B v1 Documentation](./TRACK_B_V1.md) - Detailed Track-B architecture and hyperparameters
- [README](../README.md) - Project overview and other experiment guides
- [run_track_a.sh](../scripts/run_track_a.sh) - Underlying training wrapper
