#!/usr/bin/env bash
###############################################################################
# SaladCloud Deployment Script for ECD (Embedding Compression DB)
#
# This script runs inside a SaladCloud container to:
#   1. Set up the environment
#   2. Clone/sync the project
#   3. Prepare data
#   4. Run Track-B experiments
#   5. Upload results to cloud storage (S3/GCS/Azure)
#
# Usage (inside container):
#   bash scripts/saladcloud_deploy.sh [OPTIONS]
#
# Options:
#   --skip-setup      Skip environment setup
#   --skip-data       Skip data preparation
#   --quick           Quick mode (10 steps)
#   --dry-run         Show commands without executing
#   --upload          Upload results to cloud storage when done
#
# Environment Variables (set in SaladCloud container config):
#   REPO_URL          Git repository URL
#   BRANCH            Git branch (default: main)
#   TRACK_B_STEPS     Training steps (default: 20000)
#   TRACK_B_SEEDS     Seeds to run (default: 0,1,2)
#   RESULTS_BUCKET    S3/GCS bucket for results upload
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for S3 upload)
###############################################################################
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()   { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
die()   { error "$*"; exit 1; }

###############################################################################
# Configuration
###############################################################################
: "${REPO_URL:=}"
: "${BRANCH:=main}"
: "${WORK_DIR:=/app/ecd}"
: "${SKIP_SETUP:=false}"
: "${SKIP_DATA:=false}"
: "${QUICK_MODE:=false}"
: "${DRY_RUN:=false}"
: "${UPLOAD_RESULTS:=false}"

# Track-B settings
: "${TRACK_B_STEPS:=20000}"
: "${TRACK_B_SEEDS:=0,1,2}"

# Results upload
: "${RESULTS_BUCKET:=}"
: "${RESULTS_PREFIX:=ecd-results}"

# SaladCloud specific
: "${SALAD_MACHINE_ID:=${SALAD_MACHINE_ID:-unknown}}"
: "${SALAD_CONTAINER_GROUP_ID:=${SALAD_CONTAINER_GROUP_ID:-unknown}}"

###############################################################################
# Argument Parsing
###############################################################################
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-setup)   SKIP_SETUP=true; shift ;;
            --skip-data)    SKIP_DATA=true; shift ;;
            --quick|-q)     QUICK_MODE=true; shift ;;
            --dry-run|-d)   DRY_RUN=true; shift ;;
            --upload)       UPLOAD_RESULTS=true; shift ;;
            --help|-h)      show_help; exit 0 ;;
            *)              warn "Unknown argument: $1"; shift ;;
        esac
    done
}

show_help() {
    cat <<EOF
SaladCloud Deployment Script for ECD

Usage: bash scripts/saladcloud_deploy.sh [OPTIONS]

Options:
  --skip-setup      Skip environment setup
  --skip-data       Skip data preparation
  --quick, -q       Quick mode (10 steps)
  --dry-run, -d     Show commands without executing
  --upload          Upload results to cloud storage
  --help, -h        Show this help

Environment Variables:
  REPO_URL          Git repository URL
  BRANCH            Git branch (default: main)
  TRACK_B_STEPS     Training steps (default: 20000)
  TRACK_B_SEEDS     Seeds (default: 0,1,2)
  RESULTS_BUCKET    S3/GCS bucket for results
EOF
}

###############################################################################
# System Detection
###############################################################################
detect_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        log "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 || true
        return 0
    else
        warn "No NVIDIA GPU detected"
        return 1
    fi
}

get_system_info() {
    log "=== System Information ==="
    echo "  SaladCloud Machine ID: $SALAD_MACHINE_ID"
    echo "  Container Group: $SALAD_CONTAINER_GROUP_ID"
    echo "  Hostname: $(hostname 2>/dev/null || echo 'unknown')"
    echo "  CPU Cores: $(nproc 2>/dev/null || echo 'unknown')"
    echo "  Memory: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo 'unknown')"
    detect_gpu || true
    echo ""
}

###############################################################################
# Environment Setup
###############################################################################
setup_environment() {
    log "Setting up environment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would setup environment"
        return 0
    fi
    
    # Install system packages if needed
    if command -v apt-get &>/dev/null; then
        apt-get update -qq 2>/dev/null || true
        apt-get install -y -qq git curl wget 2>/dev/null || true
    fi
    
    # Setup Python environment
    cd "$WORK_DIR"
    
    if [[ -f "uv.lock" ]] && command -v uv &>/dev/null; then
        log "Using uv to sync dependencies..."
        uv sync
        source .venv/bin/activate 2>/dev/null || true
    elif [[ -f "requirements.txt" ]]; then
        log "Using pip to install dependencies..."
        pip install -r requirements.txt
    elif [[ -f "pyproject.toml" ]]; then
        pip install -e ".[dev]"
    fi
}

###############################################################################
# Project Setup
###############################################################################
setup_project() {
    if [[ -n "$REPO_URL" ]]; then
        log "Cloning repository..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            info "[DRY RUN] Would clone $REPO_URL"
            return 0
        fi
        
        mkdir -p "$(dirname "$WORK_DIR")"
        
        if [[ -d "$WORK_DIR/.git" ]]; then
            cd "$WORK_DIR"
            git fetch origin
            git checkout "$BRANCH"
            git pull origin "$BRANCH"
        else
            git clone --branch "$BRANCH" "$REPO_URL" "$WORK_DIR"
            cd "$WORK_DIR"
        fi
    elif [[ -f "$WORK_DIR/pyproject.toml" ]]; then
        cd "$WORK_DIR"
        log "Using existing project at $WORK_DIR"
    else
        die "No REPO_URL and project not found at $WORK_DIR"
    fi
}

###############################################################################
# Data Preparation
###############################################################################
prepare_data() {
    log "Preparing data..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would prepare data"
        return 0
    fi
    
    cd "$WORK_DIR"
    [[ -f ".venv/bin/activate" ]] && source .venv/bin/activate
    
    mkdir -p logs
    python scripts/prepare_data.py --run-id salad_deploy 2>&1 | tee logs/prepare_data.log
}

###############################################################################
# Training
###############################################################################
run_training() {
    log "=========================================="
    log "Starting Track-B Experiments"
    log "=========================================="
    
    cd "$WORK_DIR"
    [[ -f ".venv/bin/activate" ]] && source .venv/bin/activate
    
    local args=()
    [[ "$QUICK_MODE" == "true" ]] && args+=("--quick")
    [[ "$DRY_RUN" == "true" ]] && args+=("--dry-run")
    
    mkdir -p logs runs
    
    export STEPS="${TRACK_B_STEPS}"
    export SEEDS="${TRACK_B_SEEDS}"
    
    log "Configuration:"
    log "  Steps: $STEPS"
    log "  Seeds: $SEEDS"
    log "  Quick Mode: $QUICK_MODE"
    log ""
    
    local start_time
    start_time=$(date +%s)
    
    bash scripts/run_track_b.sh "${args[@]}" 2>&1 | tee "logs/track_b_$(date +%Y%m%d_%H%M%S).log"
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log ""
    log "Training completed in $((duration / 3600))h $((duration % 3600 / 60))m $((duration % 60))s"
}

###############################################################################
# Results Upload
###############################################################################
upload_results() {
    if [[ -z "$RESULTS_BUCKET" ]]; then
        warn "RESULTS_BUCKET not set, skipping upload"
        return 0
    fi
    
    log "Uploading results to $RESULTS_BUCKET..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would upload to $RESULTS_BUCKET"
        return 0
    fi
    
    cd "$WORK_DIR"
    
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local dest_prefix="${RESULTS_PREFIX}/${SALAD_MACHINE_ID}/${timestamp}"
    
    # Detect cloud provider and upload
    if [[ "$RESULTS_BUCKET" == s3://* ]]; then
        # AWS S3
        if command -v aws &>/dev/null; then
            aws s3 cp summary.csv "${RESULTS_BUCKET}/${dest_prefix}/summary.csv" || true
            aws s3 cp summary.txt "${RESULTS_BUCKET}/${dest_prefix}/summary.txt" || true
            aws s3 sync runs/ "${RESULTS_BUCKET}/${dest_prefix}/runs/" --exclude "*.pt" || true
            log "Results uploaded to S3"
        else
            warn "AWS CLI not found, cannot upload to S3"
        fi
    elif [[ "$RESULTS_BUCKET" == gs://* ]]; then
        # Google Cloud Storage
        if command -v gsutil &>/dev/null; then
            gsutil cp summary.csv "${RESULTS_BUCKET}/${dest_prefix}/summary.csv" || true
            gsutil cp summary.txt "${RESULTS_BUCKET}/${dest_prefix}/summary.txt" || true
            gsutil -m rsync -r -x ".*\.pt$" runs/ "${RESULTS_BUCKET}/${dest_prefix}/runs/" || true
            log "Results uploaded to GCS"
        else
            warn "gsutil not found, cannot upload to GCS"
        fi
    elif [[ "$RESULTS_BUCKET" == https://*.blob.core.windows.net/* ]]; then
        # Azure Blob Storage
        if command -v azcopy &>/dev/null; then
            azcopy copy summary.csv "${RESULTS_BUCKET}/${dest_prefix}/summary.csv" || true
            azcopy copy summary.txt "${RESULTS_BUCKET}/${dest_prefix}/summary.txt" || true
            azcopy sync runs/ "${RESULTS_BUCKET}/${dest_prefix}/runs/" --exclude-pattern "*.pt" || true
            log "Results uploaded to Azure Blob"
        else
            warn "azcopy not found, cannot upload to Azure"
        fi
    else
        warn "Unknown bucket format: $RESULTS_BUCKET"
    fi
}

###############################################################################
# Heartbeat / Health Check
###############################################################################
start_heartbeat() {
    # SaladCloud health check endpoint
    if [[ -n "${SALAD_HEALTH_CHECK_PORT:-}" ]]; then
        log "Starting health check server on port $SALAD_HEALTH_CHECK_PORT..."
        
        # Simple HTTP health check using Python
        python3 -c "
import http.server
import threading

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')
    def log_message(self, format, *args):
        pass

server = http.server.HTTPServer(('0.0.0.0', ${SALAD_HEALTH_CHECK_PORT}), HealthHandler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
print('Health check server started')
" &
    fi
}

###############################################################################
# Main
###############################################################################
main() {
    parse_args "$@"
    
    log "=========================================="
    log "SaladCloud Deploy - ECD Project"
    log "=========================================="
    
    get_system_info
    start_heartbeat
    
    # Setup
    setup_project
    
    if [[ "$SKIP_SETUP" != "true" ]]; then
        setup_environment
    fi
    
    # Data
    if [[ "$SKIP_DATA" != "true" ]]; then
        prepare_data
    fi
    
    # Training
    run_training
    
    # Upload
    if [[ "$UPLOAD_RESULTS" == "true" ]]; then
        upload_results
    fi
    
    log ""
    log "=========================================="
    log "Deployment Complete!"
    log "=========================================="
    
    if [[ -f "$WORK_DIR/summary.txt" ]]; then
        cat "$WORK_DIR/summary.txt"
    fi
}

main "$@"
