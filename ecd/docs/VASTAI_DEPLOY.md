# vast.ai Deployment Guide

This guide covers deploying the ECD (Embedding Compression DB) project to [vast.ai](https://vast.ai) GPU instances for running Track-B experiments.

## Quick Start

### Option 1: Manual SSH Deployment

1. **Rent a GPU instance** on [vast.ai](https://vast.ai/console/create/)
   - Recommended: RTX 4090 (24GB), RTX 3090 (24GB), or A100 (40/80GB)
   - Minimum: 16GB VRAM, 32GB RAM, 50GB disk
   - Select a PyTorch image (e.g., `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`)

2. **SSH into the instance** and run:

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/ecd.git
cd ecd
bash scripts/vastai_deploy.sh
```

### Option 2: Onstart Script (Automatic)

When creating an instance, paste this into the "On-start Script" field:

```bash
#!/bin/bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/ecd.git
cd ecd
bash scripts/vastai_deploy.sh
```

### Option 3: Programmatic Launch (Python)

```bash
# Install vast.ai CLI
pip install vastai
vastai set api-key YOUR_API_KEY

# Launch instance
python scripts/vastai_launch.py --gpu-name "RTX 4090" --disk 50
```

## Deployment Scripts

### `scripts/vastai_deploy.sh`

Main deployment script that handles:
- Environment setup (Python 3.11, uv, dependencies)
- Docker services (Qdrant, PostgreSQL) if available
- Data preparation
- Running Track-B experiments

```bash
# Full deployment
bash scripts/vastai_deploy.sh

# Quick test (10 steps)
bash scripts/vastai_deploy.sh --quick

# Skip environment setup (already configured)
bash scripts/vastai_deploy.sh --skip-setup

# Skip data prep (already prepared)
bash scripts/vastai_deploy.sh --skip-data

# Dry run (show commands without executing)
bash scripts/vastai_deploy.sh --dry-run

# Custom repo/branch
bash scripts/vastai_deploy.sh --repo-url https://github.com/user/ecd.git --branch feature-x
```

### `scripts/vastai_onstart.sh`

Minimal bootstrap script for vast.ai's onstart field. Customize the variables at the top:

```bash
REPO_URL="https://github.com/YOUR_USERNAME/ecd.git"
BRANCH="main"
TRACK_B_STEPS=20000
TRACK_B_SEEDS="0,1,2"
QUICK_MODE="false"
```

### `scripts/vastai_launch.py`

Programmatic instance launcher using vast.ai API:

```bash
# Search for instances only
python scripts/vastai_launch.py --min-gpu-ram 24 --search-only

# Launch with specific GPU
python scripts/vastai_launch.py --gpu-name "RTX 4090" --disk 50

# Quick test run
python scripts/vastai_launch.py --quick --max-price 0.50

# Dry run
python scripts/vastai_launch.py --dry-run
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACK_B_STEPS` | `20000` | Training steps per experiment |
| `TRACK_B_SEEDS` | `0,1,2` | Seeds to run (comma-separated) |
| `SKIP_SETUP` | `false` | Skip environment setup |
| `SKIP_DATA` | `false` | Skip data preparation |
| `QUICK_MODE` | `false` | Run with 10 steps (testing) |

### Instance Recommendations

| GPU | VRAM | Estimated Time (20k steps) | Cost/hr |
|-----|------|---------------------------|---------|
| RTX 4090 | 24GB | ~2-3 hours | $0.30-0.50 |
| RTX 3090 | 24GB | ~3-4 hours | $0.20-0.40 |
| A100 40GB | 40GB | ~1-2 hours | $1.00-2.00 |
| A100 80GB | 80GB | ~1-2 hours | $1.50-3.00 |

### Disk Space

- Minimum: 30GB
- Recommended: 50GB (for multiple runs)
- With full datasets: 100GB+

## Monitoring

### During Training

```bash
# Attach to monitoring tmux session
tmux attach -t ecd_monitor

# Or manually watch GPU usage
watch -n 5 nvidia-smi

# Tail logs
tail -f logs/*.log

# View latest run logs
ls -la runs/
```

### After Completion

Results are saved to:
- `summary.csv` - Machine-readable results
- `summary.txt` - Human-readable summary with top runs
- `runs/` - Individual run directories with checkpoints and logs

## Retrieving Results

### Via SSH/SCP

```bash
# Get SSH connection info
vastai show instance YOUR_INSTANCE_ID

# SCP results
scp -P PORT root@HOST:/workspace/ecd/summary.csv ./
scp -rP PORT root@HOST:/workspace/ecd/runs/ ./
```

### Via vast.ai CLI

```bash
# Download specific file
vastai copy YOUR_INSTANCE_ID:/workspace/ecd/summary.csv ./

# Download all results
vastai copy YOUR_INSTANCE_ID:/workspace/ecd/runs/ ./results/
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `TRACK_B_QUEUE_SIZE` or `TRACK_B_NEG_M`
   - Use AMP (automatic mixed precision)
   
2. **Slow data loading**
   - Ensure data is prepared before training
   - Check disk I/O with `iostat`

3. **Instance terminated unexpectedly**
   - Check vast.ai console for billing issues
   - Enable instance persistence if available

4. **Python/dependency issues**
   - Run with `--skip-setup` if env is already configured
   - Check Python version: `python --version` (needs 3.11+)

### Logs Location

```
/workspace/ecd/
├── logs/
│   ├── prepare_data.log
│   ├── track_b_YYYYMMDD_HHMMSS.log
│   └── ...
├── runs/
│   └── trackB_v1_*/
│       └── logs/
└── summary.txt
```

## Cost Optimization

1. **Use spot/interruptible instances** when available
2. **Start with `--quick`** to verify setup before full runs
3. **Use checkpoints** to resume interrupted training
4. **Destroy instance immediately** after retrieving results

## Example Workflows

### Full Track-B Experiment Suite

```bash
# Launch instance
python scripts/vastai_launch.py \
    --gpu-name "RTX 4090" \
    --disk 100 \
    --repo-url https://github.com/user/ecd.git

# Monitor progress
vastai logs INSTANCE_ID -f

# Download results when done
scp -P PORT root@HOST:/workspace/ecd/summary.csv ./results/
vastai destroy instance INSTANCE_ID
```

### Quick Validation Run

```bash
# Quick test to verify everything works
bash scripts/vastai_deploy.sh --quick --skip-data

# If successful, run full suite
bash scripts/vastai_deploy.sh --skip-setup
```

### Resume After Interruption

```bash
# SSH back into instance
vastai ssh-url INSTANCE_ID

# Resume with existing data
cd /workspace/ecd
bash scripts/vastai_deploy.sh --skip-setup --skip-data
```
