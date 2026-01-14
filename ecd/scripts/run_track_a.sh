#!/usr/bin/env bash
set -euo pipefail


die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARN: $*" >&2; }

have() { command -v "$1" >/dev/null 2>&1; }

run_cmd() {
  local log_path="$1"; shift
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "+ $*"
    return 0
  fi
  echo "+ $*"
  mkdir -p "$(dirname "$log_path")"
  "$@" 2>&1 | tee -a "$log_path"
}

now_ts() { date +"%Y%m%d_%H%M%S"; }

cpu_cores() {
  if have nproc; then nproc
  elif have sysctl; then sysctl -n hw.ncpu
  else echo "unknown"
  fi
}

os_name() {
  uname -a 2>/dev/null || echo "unknown"
}

git_hash() {
  if have git && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD 2>/dev/null || echo "unknown"
  else
    echo "unknown"
  fi
}

gpu_info() {
  if have nvidia-smi; then
    nvidia-smi -L 2>/dev/null || true
  else
    echo "nvidia-smi not found"
  fi
}

probe_help_has() {
  local base="$1"
  local needle="$2"
  local out
  set +e
  out=$($base --help 2>/dev/null)
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    return 1
  fi
  echo "$out" | awk -v n="$needle" 'index($0,n)>0 {found=1} END{exit found?0:1}'
}

ensure_repo_root() {
  [[ -d "scripts" ]] || warn "No ./scripts dir found; are you running from repo root?"
}

detect_sweep_quick() {
  if [[ -f "scripts/sweep_quick_hyper.py" ]]; then
    echo "python scripts/sweep_quick_hyper.py"
  else
    echo ""
  fi
}

detect_sweep_hardneg() {
  if [[ -f "scripts/sweep_hardneg.py" ]]; then
    echo "python scripts/sweep_hardneg.py"
  else
    echo ""
  fi
}

detect_train_entrypoint() {
  if [[ -f "train.py" ]]; then
    echo "python train.py"
    return 0
  fi
  if [[ -f "scripts/train.py" ]]; then
    echo "python scripts/train.py"
    return 0
  fi
  if [[ -f "scripts/train_student.py" ]]; then
    echo "python scripts/train_student.py"
    return 0
  fi
  echo ""
}

apply_preset() {
  local preset="${1:-}"
  case "$preset" in
    ""|help|smoke|single|sweep|sweepA|resume)
      return 0
      ;;
    a1)
      : "${STEPS:=20000}"
      : "${LR_SCHEDULE:=cosine}"
      : "${WARMUP_STEPS:=1000}"
      : "${AMP:=bf16}"
      : "${TARGET:=0.70}"
      return 0
      ;;
    a2)
      : "${STEPS:=50000}"
      : "${LR_SCHEDULE:=cosine}"
      : "${WARMUP_STEPS:=1000}"
      : "${AMP:=bf16}"
      : "${TARGET:=0.70}"
      return 0
      ;;
    smoke_preset|smokepreset)
      : "${STEPS:=200}"
      : "${SEEDS:=0}"
      : "${MAX_RUNS:=1}"
      : "${EVAL_EVERY:=0}"
      : "${SAVE_EVERY:=200}"
      : "${AMP:=none}"
      return 0
      ;;
    *)
      die "Unknown preset '$preset'. Try: help"
      ;;
  esac
}

CMD="${1:-help}"
PRESET_OR_CMD="${1:-help}"

if [[ "$PRESET_OR_CMD" =~ ^(a1|a2|smoke_preset|smokepreset)$ ]]; then
  apply_preset "$PRESET_OR_CMD"
  CMD="${2:-help}"
  shift 1
else
  apply_preset "$PRESET_OR_CMD" || true
fi

: "${DRY_RUN:=0}"
: "${RUN_ID:=quickA_$(now_ts)}"

DEFAULT_CONFIG="configs/sweep/quick_hyper.yaml"
: "${CONFIG:=$DEFAULT_CONFIG}"

: "${STEPS:=5000}"
: "${LR:=}"
: "${LR_SCHEDULE:=cosine}"
: "${WARMUP_STEPS:=0}"
: "${AMP:=none}"
: "${SAVE_EVERY:=2000}"
: "${EVAL_EVERY:=0}"
: "${SEEDS:=0,1,2}"
: "${MAX_RUNS:=0}"
: "${TARGET:=0.70}"
: "${RESUME:=}"

: "${MIX_RANDOM:=0.2}"
: "${TAIL_TO:=50}"
: "${NUM_POSITIVES:=8}"
: "${BS:=}"
: "${BATCH_SIZE:=${BS}}"

OUT_DIR="runs/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
RESULTS_DIR="${OUT_DIR}/results"
CONFIGS_DIR="${OUT_DIR}/configs"
MANIFEST="${OUT_DIR}/run_manifest.txt"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CONFIGS_DIR"

ensure_repo_root

have python || die "python not found in PATH"

if [[ ! -f "$CONFIG" ]]; then
  cat >&2 <<EOF
Config file not found: $CONFIG

How to set:
  CONFIG=path/to/config.yaml $0 sweep

Default expected at:
  $DEFAULT_CONFIG
EOF
  exit 1
fi

cp -f "$CONFIG" "${CONFIGS_DIR}/selected_config.yaml"

{
  echo "run_id: $RUN_ID"
  echo "timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "git_commit: $(git_hash)"
  echo "hostname: $(uname -n 2>/dev/null || echo unknown)"
  echo "os: $(os_name)"
  echo "cpu_cores: $(cpu_cores)"
  echo "gpu: $(gpu_info)"
  echo "config: $CONFIG"
  echo "out_dir: $OUT_DIR"
  echo "env:"
  echo "  STEPS=$STEPS"
  echo "  LR=${LR:-<unset>}"
  echo "  LR_SCHEDULE=$LR_SCHEDULE"
  echo "  WARMUP_STEPS=$WARMUP_STEPS"
  echo "  AMP=$AMP"
  echo "  SAVE_EVERY=$SAVE_EVERY"
  echo "  EVAL_EVERY=$EVAL_EVERY"
  echo "  SEEDS=$SEEDS"
  echo "  MAX_RUNS=$MAX_RUNS"
  echo "  TARGET=$TARGET"
  echo "  RESUME=${RESUME:-<unset>}"
  echo "  MIX_RANDOM=$MIX_RANDOM"
  echo "  TAIL_TO=$TAIL_TO"
  echo "  NUM_POSITIVES=$NUM_POSITIVES"
  echo "  BATCH_SIZE=${BATCH_SIZE:-<unset>}"
} >"$MANIFEST"

build_sweep_overrides() {
  local -a ovs=()

  ovs+=("--override" "sweep.target_vs_teacher_recall10=${TARGET}")

  ovs+=("--override" "sweep.grid.steps=[${STEPS}]")
  if [[ -n "${LR}" ]]; then
    ovs+=("--override" "sweep.grid.lr=[${LR}]")
  fi
  ovs+=("--override" "sweep.grid.lr_schedule=[${LR_SCHEDULE}]")
  ovs+=("--override" "sweep.grid.warmup_steps=[${WARMUP_STEPS}]")
  ovs+=("--override" "sweep.grid.amp=[${AMP}]")

  ovs+=("--override" "train.save_every_steps=${SAVE_EVERY}")
  ovs+=("--override" "train.eval_every_steps=${EVAL_EVERY}")

  ovs+=("--override" "sweep.grid.mix_random_ratio=[${MIX_RANDOM}]")
  ovs+=("--override" "sweep.grid.tail_to=[${TAIL_TO}]")
  ovs+=("--override" "sweep.grid.num_positives=[${NUM_POSITIVES}]")

  if [[ -n "${BATCH_SIZE}" ]]; then
    ovs+=("--override" "sweep.grid.batch_size=[${BATCH_SIZE}]")
  fi

  printf "%s\0" "${ovs[@]}"
}

build_sweepA_overrides_for_grid() {
  local steps_list="$1"
  local sched_list="$2"
  local warmup_list="$3"
  local amp_list="$4"

  local -a ovs=()
  ovs+=("--override" "sweep.target_vs_teacher_recall10=${TARGET}")

  ovs+=("--override" "sweep.grid.steps=${steps_list}")
  if [[ -n "${LR}" ]]; then
    ovs+=("--override" "sweep.grid.lr=[${LR}]")
  fi
  ovs+=("--override" "sweep.grid.lr_schedule=${sched_list}")
  ovs+=("--override" "sweep.grid.warmup_steps=${warmup_list}")
  ovs+=("--override" "sweep.grid.amp=${amp_list}")

  ovs+=("--override" "train.save_every_steps=${SAVE_EVERY}")
  ovs+=("--override" "train.eval_every_steps=${EVAL_EVERY}")

  ovs+=("--override" "sweep.grid.mix_random_ratio=[${MIX_RANDOM}]")
  ovs+=("--override" "sweep.grid.tail_to=[${TAIL_TO}]")
  ovs+=("--override" "sweep.grid.num_positives=[${NUM_POSITIVES}]")

  if [[ -n "${BATCH_SIZE}" ]]; then
    ovs+=("--override" "sweep.grid.batch_size=[${BATCH_SIZE}]")
  fi

  printf "%s\0" "${ovs[@]}"
}

build_single_overrides() {
  local -a ovs=()
  ovs+=("--override" "train.steps=${STEPS}")
  ovs+=("--override" "train.train_steps=${STEPS}")
  if [[ -n "${LR}" ]]; then
    ovs+=("--override" "train.lr=${LR}")
  fi
  ovs+=("--override" "train.lr_schedule=${LR_SCHEDULE}")
  ovs+=("--override" "train.warmup_steps=${WARMUP_STEPS}")
  ovs+=("--override" "train.amp=${AMP}")
  ovs+=("--override" "train.save_every_steps=${SAVE_EVERY}")
  ovs+=("--override" "train.eval_every_steps=${EVAL_EVERY}")

  ovs+=("--override" "train.hard_negative.mix_random_ratio=${MIX_RANDOM}")
  ovs+=("--override" "train.hard_negative.tail_to=${TAIL_TO}")
  ovs+=("--override" "train.rank.multi_positive.num_positives=${NUM_POSITIVES}")
  ovs+=("--override" "train.rank.multi_positive.enabled=true")
  ovs+=("--override" "train.hard_negative.enabled=true")
  ovs+=("--override" "train.hard_negative.mode=teacher_tail")

  if [[ -n "${RESUME}" ]]; then
    ovs+=("--override" "resume=${RESUME}")
    ovs+=("--override" "train.resume=${RESUME}")
    ovs+=("--override" "train.ckpt_path=${RESUME}")
  fi

  printf "%s\0" "${ovs[@]}"
}

print_help() {
  cat <<EOF
Usage:
  $0 help
  $0 smoke
  $0 sweep
  $0 sweepA
  $0 single
  $0 resume

Presets:
  $0 a1 sweepA
  $0 a2 sweepA

Environment:
  CONFIG=... RUN_ID=... DRY_RUN=1 ...

Examples (6):
  1) Smoke test:
     $0 smoke

  2) Sweep with custom RUN_ID:
     RUN_ID=quickA_\$(date +%Y%m%d_%H%M%S) $0 sweep

  3) Default Track-A grid sweep:
     $0 sweepA

  4) Sweep setting STEPS/LR_SCHEDULE/WARMUP_STEPS:
     STEPS=20000 LR_SCHEDULE=cosine WARMUP_STEPS=1000 AMP=bf16 $0 sweep

  5) Single run with resume:
     RESUME=results/<old_run_id>/checkpoints/student_best.pt $0 single

  6) Quick-hyper knob overrides:
     MIX_RANDOM=0.3 TAIL_TO=100 NUM_POSITIVES=8 $0 sweep

Notes:
- sweep_quick_hyper only supports --config/--run_id/--seed/--override.
  Track-A knobs are passed via repeated --override keys into sweep.grid.*.
- Outputs/logs for this wrapper go under: runs/<RUN_ID>/
EOF
}

cmd_sweep() {
  local sweep_entry
  sweep_entry="$(detect_sweep_quick)"
  [[ -n "$sweep_entry" ]] || die "scripts/sweep_quick_hyper.py not found; cannot run sweep"

  local -a argv=()
  argv+=($sweep_entry)
  argv+=("--config" "$CONFIG")

  if probe_help_has "$sweep_entry" "--run_id"; then
    argv+=("--run_id" "$RUN_ID")
  elif probe_help_has "$sweep_entry" "--run-id"; then
    argv+=("--run-id" "$RUN_ID")
  fi

  local IFS=','
  local -a seed_arr
  read -r -a seed_arr <<<"$SEEDS"

  local -a ov=()
  while IFS= read -r -d '' item; do ov+=("$item"); done < <(build_sweep_overrides)

  if [[ "${MAX_RUNS}" != "0" ]]; then
    ov+=("--override" "sweep.max_runs=${MAX_RUNS}")
  fi

  local seed
  local runs_done=0
  for seed in "${seed_arr[@]}"; do
    local -a cmd=("${argv[@]}")
    if probe_help_has "$sweep_entry" "--seed"; then
      cmd+=("--seed" "$seed")
    else
      cmd+=("--override" "sweep.seed=${seed}")
    fi
    cmd+=("${ov[@]}")

    run_cmd "${LOG_DIR}/sweep_seed${seed}.log" "${cmd[@]}"

    runs_done=$((runs_done + 1))
    if [[ "${MAX_RUNS}" -gt 0 && "${runs_done}" -ge "${MAX_RUNS}" ]]; then
      echo "Reached MAX_RUNS=${MAX_RUNS}; stopping."
      break
    fi
  done
}

cmd_sweepA() {
  local sweep_entry
  sweep_entry="$(detect_sweep_quick)"
  [[ -n "$sweep_entry" ]] || die "scripts/sweep_quick_hyper.py not found; cannot run sweepA"

  local steps_list="[5000,20000,50000]"
  local sched_list="[cosine,constant]"
  local warmup_list="[0,500,1000]"

  local amp_list="[none,bf16]"

  local -a argv=()
  argv+=($sweep_entry)
  argv+=("--config" "$CONFIG")

  if probe_help_has "$sweep_entry" "--run_id"; then
    argv+=("--run_id" "$RUN_ID")
  elif probe_help_has "$sweep_entry" "--run-id"; then
    argv+=("--run-id" "$RUN_ID")
  fi

  local -a ov=()
  while IFS= read -r -d '' item; do ov+=("$item"); done < <(build_sweepA_overrides_for_grid "$steps_list" "$sched_list" "$warmup_list" "$amp_list")

  local IFS=','
  local -a seed_arr
  read -r -a seed_arr <<<"$SEEDS"

  local seed
  local runs_done=0
  for seed in "${seed_arr[@]}"; do
    local -a cmd=("${argv[@]}")
    if probe_help_has "$sweep_entry" "--seed"; then
      cmd+=("--seed" "$seed")
    else
      cmd+=("--override" "sweep.seed=${seed}")
    fi
    cmd+=("${ov[@]}")

    run_cmd "${LOG_DIR}/sweepA_seed${seed}.log" "${cmd[@]}"

    runs_done=$((runs_done + 1))
    if [[ "${MAX_RUNS}" -gt 0 && "${runs_done}" -ge "${MAX_RUNS}" ]]; then
      echo "Reached MAX_RUNS=${MAX_RUNS}; stopping."
      break
    fi
  done
}

cmd_smoke() {
  STEPS=200
  SEEDS="0"
  MAX_RUNS=1
  AMP="none"
  SAVE_EVERY=200
  EVAL_EVERY=0
  cmd_sweep
}

cmd_single() {
  local train_entry
  train_entry="$(detect_train_entrypoint)"
  [[ -n "$train_entry" ]] || die "No training entrypoint detected (train.py/scripts/train.py/scripts/train_student.py)."

  local -a argv=()
  argv+=($train_entry)

  if probe_help_has "$train_entry" "--run-id"; then
    argv+=("--run-id" "$RUN_ID")
  elif probe_help_has "$train_entry" "--run_id"; then
    argv+=("--run_id" "$RUN_ID")
  fi

  local train_cfg="configs/train/default.yaml"
  if [[ -f "configs/train/default.yaml" ]]; then
    if probe_help_has "$train_entry" "--config"; then
      argv+=("--config" "$train_cfg")
    elif probe_help_has "$train_entry" "--train-config"; then
      argv+=("--train-config" "$train_cfg")
    fi
  else
    warn "configs/train/default.yaml not found; running training without explicit train config flag."
  fi

  local -a ov=()
  while IFS= read -r -d '' item; do ov+=("$item"); done < <(build_single_overrides)

  argv+=("${ov[@]}")

  run_cmd "${LOG_DIR}/single.log" "${argv[@]}"
}

cmd_resume() {
  if [[ -z "${RESUME}" ]]; then
    die "RESUME is empty. Set RESUME=/path/to/checkpoint and rerun: RESUME=... $0 resume"
  fi
  cmd_single
}

case "$CMD" in
  help|-h|--help)
    print_help
    ;;
  smoke)
    cmd_smoke
    ;;
  sweep)
    cmd_sweep
    ;;
  sweepA)
    cmd_sweepA
    ;;
  single)
    cmd_single
    ;;
  resume)
    cmd_resume
    ;;
  *)
    die "Unknown command '$CMD'. Try: $0 help"
    ;;
esac
