#!/usr/bin/env bash
set -euo pipefail

echo "=== Track-B v1 Smoke Test ==="
echo ""
echo "Testing Track-B v1 listwise distillation with minimal configuration."
echo ""

RUN_ID="trackb_smoke_$(date +%Y%m%d_%H%M%S)"

TRACK_B_ENABLE=true \
TRACK_B_TAU=0.07 \
TRACK_B_K=50 \
TRACK_B_M=1024 \
TRACK_B_QUEUE_SIZE=32000 \
TRACK_B_FALSE_NEG_FILTER_MODE=threshold \
TRACK_B_FALSE_NEG_THRESHOLD=0.8 \
TRACK_B_MIX_LAMBDA=1.0 \
TRACK_B_QUEUE_CPU_FALLBACK=true \
TRACK_B_USE_SKIP=true \
TRACK_B_ALPHA_INIT=0.1 \
STEPS=200 \
SEEDS=0 \
MAX_RUNS=1 \
SAVE_EVERY=100 \
EVAL_EVERY=0 \
DRY_RUN=1 \
bash scripts/run_track_a.sh single

echo ""
echo "=== Smoke Test Completed ==="
echo ""
echo "To run for real, remove DRY_RUN=1 from the command above."
echo "Example:"
echo "  TRACK_B_ENABLE=true TRACK_B_K=50 TRACK_B_M=1024 bash scripts/run_track_a.sh single"
echo ""
echo "Available Track-B environment variables:"
echo "  TRACK_B_ENABLE=false|true (default: false)"
echo "  TRACK_B_TAU=0.05|0.07|0.10 (default: 0.07)"
echo "  TRACK_B_K=25|50|100 (default: 50, number of teacher top-k positives)"
echo "  TRACK_B_M=512|1024|2048 (default: 1024, number of queue negatives)"
echo "  TRACK_B_QUEUE_SIZE=32000|64000 (default: 32000)"
echo "  TRACK_B_FALSE_NEG_FILTER_MODE=none|threshold|top_percent (default: threshold)"
echo "  TRACK_B_FALSE_NEG_THRESHOLD=0.7|0.8|0.9 (default: 0.8)"
echo "  TRACK_B_FALSE_NEG_TOP_PERCENT=0.01|0.02|0.05 (default: 0.02)"
echo "  TRACK_B_MIX_LAMBDA=0.5|1.0 (default: 1.0, mix lambda for Track-A vs Track-B)"
echo "  TRACK_B_QUEUE_CPU_FALLBACK=true|false (default: true)"
echo "  TRACK_B_USE_SKIP=true|false (default: true)"
echo "  TRACK_B_ALPHA_INIT=0.1 (default: 0.1, learnable skip alpha)"
