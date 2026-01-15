#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARN: $*" >&2; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

check_repo_root() {
  [[ -d "scripts" ]] || die "Not in repo root: scripts/ directory not found"
  [[ -f "scripts/run_track_a.sh" ]] || die "scripts/run_track_a.sh not found"
}

get_timestamp() {
  date +"%Y%m%d_%H%M%S"
}

generate_run_id() {
  local exp_name="$1"
  local steps="$2"
  local seeds="$3"
  local tau="$4"
  local neg_m="$5"
  local queue_size="$6"
  local ts="$(get_timestamp)"

  echo "trackB_v1_${exp_name}_s${steps}_seeds${seeds}_tau${tau}_m${neg_m}_q${queue_size}_${ts}"
}

parse_metrics() {
  local run_dir="$1"

  if [[ ! -d "$run_dir/logs" ]]; then
    echo '{"status":"missing_logs","vs_teacher_recall10":null,"vs_teacher_ndcg10":null}'
    return
  fi

  if [[ ! -f "scripts/extract_trackb_metrics.py" ]]; then
    echo '{"status":"missing_helper","vs_teacher_recall10":null,"vs_teacher_ndcg10":null}'
    return
  fi

  python scripts/extract_trackb_metrics.py "$run_dir/logs" || \
    echo '{"status":"extract_error","vs_teacher_recall10":null,"vs_teacher_ndcg10":null}'
}

run_experiment() {
  local exp_name="$1"
  local steps="$2"
  local seeds="$3"
  local tau="$4"

  local neg_m="${TRACK_B_NEG_M:-4096}"
  local queue_size="${TRACK_B_QUEUE_SIZE:-12582912}"  # 12M queue (~48GB VRAM) for 60GB GPUs
  local lr="0.0005"
  local lr_schedule="cosine"
  local warmup="2000"
  local amp="${AMP:-none}"
  local dim="128"
  local k="50"
  local false_neg_mode="threshold"
  local false_neg_threshold="0.8"
  local batch_size="${BATCH_SIZE:-65536}"  # Start large, auto-reduces on OOM

  local run_id
  run_id="$(generate_run_id "$exp_name" "$steps" "$seeds" "$tau" "$neg_m" "$queue_size")"
  local run_dir="runs/${run_id}"

  if [[ "$SKIP_EXISTING" == "true" ]] && [[ -d "$run_dir" ]]; then
    local metrics_json
    metrics_json="$(parse_metrics "$run_dir")"
    local status
    status="$(echo "$metrics_json" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "unknown")"

    if [[ "$status" == "success" ]]; then
      local recall10
      recall10="$(echo "$metrics_json" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('vs_teacher_recall10','null'))" 2>/dev/null || echo "null")"
      log "[SKIP] Run $run_id already completed with recall@10=$recall10"
      echo "$run_id|$status|$recall10|null"
      return
    fi
  fi

  log "[RUN] Starting: $run_id"
  log "    Config: steps=$steps, seeds=$seeds, tau=$tau, neg_m=$neg_m"

  if [[ "$DRY_RUN" == "true" ]]; then
    log "    [DRY RUN] Would execute:"
    log "      TRACK_B_ENABLE=true TRACK_B_DIM=$dim TRACK_B_K=$k TRACK_B_NEG_M=$neg_m"
    log "      TRACK_B_QUEUE_SIZE=$queue_size TRACK_B_TAU=$tau TRACK_B_FALSE_NEG_MODE=$false_neg_mode"
    log "      TRACK_B_FALSE_NEG_THRESHOLD=$false_neg_threshold STEPS=$steps SEEDS=$seeds"
    log "      LR=$lr LR_SCHEDULE=$lr_schedule WARMUP_STEPS=$warmup AMP=$amp"
    log "      BATCH_SIZE=$batch_size (auto-reduces on OOM)"
    log "      RUN_ID=$run_id bash scripts/run_track_a.sh sweep"
    log ""
    echo "$run_id|dry_run|null/null"
    return
  fi

  local env_vars=(
    "TRACK_B_ENABLE=true"
    "TRACK_B_DIM=$dim"
    "TRACK_B_K=$k"
    "TRACK_B_NEG_M=$neg_m"
    "TRACK_B_QUEUE_SIZE=$queue_size"
    "TRACK_B_TAU=$tau"
    "TRACK_B_FALSE_NEG_MODE=$false_neg_mode"
    "TRACK_B_FALSE_NEG_THRESHOLD=$false_neg_threshold"
    "STEPS=$steps"
    "SEEDS=$seeds"
    "LR=$lr"
    "LR_SCHEDULE=$lr_schedule"
    "WARMUP_STEPS=$warmup"
    "AMP=$amp"
    "BATCH_SIZE=$batch_size"
    "RUN_ID=$run_id"
    "SAVE_EVERY=2000"
    "EVAL_EVERY=1000"
  )

  local result_status="failed"
  local recall10="null"
  local ndcg10="null"

  if env "${env_vars[@]}" bash scripts/run_track_a.sh sweep 2>&1 | tee -a /dev/tty; then
    local metrics_json
    metrics_json="$(parse_metrics "$run_dir")"
    result_status="$(echo "$metrics_json" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('status','failed'))" 2>/dev/null || echo "parse_error")"

    if [[ "$result_status" == "success" ]]; then
      recall10="$(echo "$metrics_json" | python3 -c "import sys, json; d=json.load(sys.stdin); v=d.get('vs_teacher_recall10'); print(f'{v:.6f}' if v is not None else 'null')" 2>/dev/null || echo "null")"
      ndcg10="$(echo "$metrics_json" | python3 -c "import sys, json; d=json.load(sys.stdin); v=d.get('vs_teacher_ndcg10'); print(f'{v:.6f}' if v is not None else 'null')" 2>/dev/null || echo "null")"
      log "[DONE] Run $run_id completed: status=$result_status"
      log "       vs_teacher_recall@10=$recall10"
      log "       vs_teacher_ndcg@10=$ndcg10"
    else
      log "       WARNING: Could not parse metrics (status=$result_status)"
    fi
  else
    log "[ERROR] Run $run_id failed"
  fi

  echo ""
  echo "$run_id|$result_status|$recall10|$ndcg10"
}

write_summary_csv() {
  local results_file="$1"
  local output_path="$2"
  local timestamp
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

  cat > "$output_path" <<EOF
timestamp,run_id,experiment,steps,seeds,lr,warmup,tau,k,neg_m,queue_size,false_neg_mode,threshold,status,vs_teacher_recall10,vs_teacher_ndcg10
EOF

  while IFS='|' read -r run_id status recall10 ndcg10 exp_name steps seeds lr warmup tau k neg_m queue_size false_neg_mode false_neg_threshold; do
    echo "$timestamp,$run_id,$exp_name,$steps,$seeds,$lr,$warmup,$tau,$k,$neg_m,$queue_size,$false_neg_mode,$false_neg_threshold,$status,$recall10,$ndcg10" >> "$output_path"
  done < "$results_file"
}

write_summary_txt() {
  local results_file="$1"
  local output_path="$2"
  local timestamp
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  local total_runs
  total_runs="$(wc -l < "$results_file")"

  cat > "$output_path" <<EOF
==========================================
Track-B v1 Experiment Summary
==========================================
Generated: $timestamp
Total runs: $total_runs

EOF

  echo "==========================================" >> "$output_path"
  echo "Top 3 Runs by vs_teacher_recall@10" >> "$output_path"
  echo "==========================================" >> "$output_path"

  local top3
  top3="$(
    while IFS='|' read -r run_id status recall10 ndcg10 rest; do
      if [[ "$status" == "success" ]] && [[ "$recall10" != "null" ]]; then
        echo "$recall10|$run_id|$ndcg10|$rest"
      fi
    done < "$results_file" | sort -t'|' -k1 -rn | head -3
  )"

  if [[ -z "$top3" ]]; then
    echo "No successful runs with metrics." >> "$output_path"
  else
    local rank=1
    while IFS='|' read -r recall10 run_id ndcg10 exp_name steps seeds lr warmup tau k neg_m queue_size false_neg_mode false_neg_threshold; do
      echo "" >> "$output_path"
      echo "  $run_id" >> "$output_path"
      echo "    recall@10=$recall10 ndcg@10=$ndcg10 steps=$steps tau=$tau" >> "$output_path"
      ((rank++))
    done <<< "$top3"
  fi

  echo "" >> "$output_path"
  echo "==========================================" >> "$output_path"
  echo "Average Metrics (successful runs)" >> "$output_path"
  echo "==========================================" >> "$output_path"

  local avg_recall avg_ndcg count
  avg_recall="$(
    while IFS='|' read -r run_id status recall10 rest; do
      if [[ "$status" == "success" ]] && [[ "$recall10" != "null" ]]; then
        echo "$recall10"
      fi
    done < "$results_file" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}'
  )"

  avg_ndcg="$(
    while IFS='|' read -r run_id status recall10 ndcg10 rest; do
      if [[ "$status" == "success" ]] && [[ "$ndcg10" != "null" ]]; then
        echo "$ndcg10"
      fi
    done < "$results_file" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}'
  )"

  echo "  Average vs_teacher_recall@10: $avg_recall" >> "$output_path"
  echo "  Average vs_teacher_ndcg@10: $avg_ndcg" >> "$output_path"

  echo "" >> "$output_path"
  echo "==========================================" >> "$output_path"
  echo "Failed / Incomplete Runs" >> "$output_path"
  echo "==========================================" >> "$output_path"

  local failed
  failed="$(
    while IFS='|' read -r run_id status recall10 ndcg10 rest; do
      if [[ "$status" != "success" ]] && [[ "$status" != "skipped" ]] && [[ "$status" != "dry_run" ]]; then
        echo "  $run_id - status: $status"
      fi
    done < "$results_file"
  )"

  if [[ -z "$failed" ]]; then
    echo "None - all runs successful or skipped." >> "$output_path"
  else
    echo "$failed" >> "$output_path"
  fi

  echo "" >> "$output_path"
  echo "==========================================" >> "$output_path"
  echo "Full Run List" >> "$output_path"
  echo "==========================================" >> "$output_path"

  while IFS='|' read -r run_id status recall10 ndcg10 rest; do
    echo "  $run_id - status: $status recall@10: $recall10" >> "$output_path"
  done < "$results_file"
}

main() {
  local dry_run="false"
  local quick_mode="false"
  local force_rerun="false"

  for arg in "$@"; do
    case "$arg" in
      --dry-run|-d)
        dry_run="true"
        ;;
      --quick|-q)
        quick_mode="true"
        ;;
      --force|-f)
        force_rerun="true"
        ;;
      --help|-h)
        echo "Track-B v1 Experiment Orchestration"
        echo ""
        echo "Usage: bash scripts/run_track_b.sh [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --dry-run, -d    Show commands that would be executed without running"
        echo "  --quick, -q      Quick mode (10 steps instead of full training)"
        echo "  --force, -f       Force rerun all experiments (skip existing=false)"
        echo "  --help, -h        Show this help message"
        echo ""
        echo "Experiments (run in sequence):"
        echo "  1) Sanity check: check_trackb_sanity.py (if exists)"
        echo "  2) Core run 1: 20k steps, seed0"
        echo "  3) Core run 2: 20k steps, seeds 0,1,2"
        echo "  4) Core run 3: 50k steps, seeds 0,1,2"
        echo "  5) Tuning A: 20k steps, seed0, tau=0.05"
        echo "  6) Tuning B: 20k steps, seed0, tau=0.10"
        echo ""
        echo "Output files:"
        echo "  summary.csv  - Machine-readable summary of all runs"
        echo "  summary.txt  - Human-readable summary with top runs"
        echo ""
        echo "Environment variables (passed to run_track_a.sh):"
        echo "  TRACK_B_ENABLE=true (auto-set)"
        echo "  TRACK_B_DIM=128"
        echo "  TRACK_B_K=50"
        echo "  TRACK_B_NEG_M=4096"
        echo "  TRACK_B_QUEUE_SIZE=4194304  # 4M entries (~16GB VRAM)"
        echo "  TRACK_B_TAU=0.07 (varies per experiment)"
        echo "  TRACK_B_FALSE_NEG_MODE=threshold"
        echo "  TRACK_B_FALSE_NEG_THRESHOLD=0.8"
        echo "  STEPS=20000 or 50000 (varies per experiment)"
        echo "  LR=0.0005"
        echo "  LR_SCHEDULE=cosine"
        echo "  WARMUP_STEPS=2000"
        echo "  AMP=none"
        echo ""
        echo "Batch size environment variable:"
        echo "  BATCH_SIZE=65536       Start large, auto-reduces on OOM to find max"
        echo ""
        echo "Example:"
        echo "  AMP=bf16 bash scripts/run_track_b.sh  # Uses default 65536, auto-finds optimal"
        exit 0
        ;;
    esac
  done

  local skip_existing="true"
  [[ "$force_rerun" == "true" ]] && skip_existing="false"

  export DRY_RUN="$dry_run"
  export SKIP_EXISTING="$skip_existing"

  check_repo_root
  mkdir -p runs

  log "=========================================="
  log "Track-B v1 Experiment Orchestration"
  log "=========================================="
  [[ "$dry_run" == "true" ]] && log "[DRY RUN MODE - No actual training]"
  [[ "$quick_mode" == "true" ]] && log "[QUICK MODE - Using 10 steps]"
  log ""

  log "[1/6] Sanity check..."
  if [[ -f "scripts/check_trackb_sanity.py" ]]; then
    if [[ "$dry_run" == "true" ]]; then
      log "  [DRY RUN] Would run: python scripts/check_trackb_sanity.py"
    else
      if python scripts/check_trackb_sanity.py; then
        log "  ✓ Sanity check passed"
      else
        log "  ⚠ Sanity check failed (continuing anyway)"
      fi
    fi
  else
    log "  ⚠ check_trackb_sanity.py not found, skipping"
  fi
  log ""

  local base_steps long_steps
  if [[ "$quick_mode" == "true" ]]; then
    base_steps=10
    long_steps=10
  else
    base_steps=20000
    long_steps=50000
  fi

  local results_file
  results_file="$(mktemp)"
  trap "rm -f $results_file" EXIT

  local result

  result="$(run_experiment "core_20k_seed0" "$base_steps" "0" "0.07")"
  IFS='|' read -r run_id status recall10 ndcg10 <<< "$result"
  echo "$run_id|$status|$recall10|$ndcg10|core_20k_seed0|$base_steps|0|0.0005|2000|0.07|50|1024|32768|threshold|0.8" >> "$results_file"

  result="$(run_experiment "core_20k_seeds012" "$base_steps" "0,1,2" "0.07")"
  IFS='|' read -r run_id status recall10 ndcg10 <<< "$result"
  echo "$run_id|$status|$recall10|$ndcg10|core_20k_seeds012|$base_steps|0,1,2|0.0005|2000|0.07|50|1024|32768|threshold|0.8" >> "$results_file"

  result="$(run_experiment "core_50k_seeds012" "$long_steps" "0,1,2" "0.07")"
  IFS='|' read -r run_id status recall10 ndcg10 <<< "$result"
  echo "$run_id|$status|$recall10|$ndcg10|core_50k_seeds012|$long_steps|0,1,2|0.0005|2000|0.07|50|1024|32768|threshold|0.8" >> "$results_file"

  result="$(run_experiment "tuning_tau0.05" "$base_steps" "0" "0.05")"
  IFS='|' read -r run_id status recall10 ndcg10 <<< "$result"
  echo "$run_id|$status|$recall10|$ndcg10|tuning_tau0.05|$base_steps|0|0.0005|2000|0.05|50|1024|32768|threshold|0.8" >> "$results_file"

  result="$(run_experiment "tuning_tau0.10" "$base_steps" "0" "0.10")"
  IFS='|' read -r run_id status recall10 ndcg10 <<< "$result"
  echo "$run_id|$status|$recall10|$ndcg10|tuning_tau0.10|$base_steps|0|0.0005|2000|0.10|50|1024|32768|threshold|0.8" >> "$results_file"

  log ""
  log "[FINAL] Writing summaries..."

  if [[ "$dry_run" == "true" ]]; then
    log "  [DRY RUN] Skipping summary file generation"
  else
    write_summary_csv "$results_file" "summary.csv"
    log "  ✓ summary.csv written"

    write_summary_txt "$results_file" "summary.txt"
    log "  ✓ summary.txt written"
  fi

  log ""
  log "=========================================="
  log "All experiments completed!"
  log "=========================================="
  log "Results:"
  log "  summary.csv - Machine-readable summary"
  log "  summary.txt - Human-readable summary"
  log ""
}

main "$@"
