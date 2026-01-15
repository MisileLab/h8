#!/usr/bin/env bash
###############################################################################
# vast.ai Auto Deploy Script for ECD (Embedding Compression DB)
# 
# This script automates the full deployment workflow on vast.ai instances:
#   1. Environment setup (Python, CUDA, dependencies)
#   2. Project cloning/syncing
#   3. Data preparation
#   4. Running Track-B experiments
#
# Usage:
#   On vast.ai instance (after SSH):
#     curl -sSL <your-repo-url>/scripts/vastai_deploy.sh | bash
#   
#   Or clone first, then:
#     bash scripts/vastai_deploy.sh [OPTIONS]
#
# Options:
#   --skip-setup      Skip environment setup (assume already configured)
#   --skip-data       Skip data preparation step
#   --quick           Run quick mode (10 steps instead of full training)
#   --dry-run         Show what would run without executing
#   --repo-url URL    Git repo URL to clone (default: uses current dir)
#   --branch BRANCH   Git branch to checkout (default: main)
#   --help            Show this help message
###############################################################################
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()   { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }

die() { error "$*"; exit 1; }

###############################################################################
# Configuration
###############################################################################
: "${REPO_URL:=}"
: "${BRANCH:=main}"
: "${WORK_DIR:=/workspace/ecd}"
: "${SKIP_SETUP:=false}"
: "${SKIP_DATA:=false}"
: "${QUICK_MODE:=false}"
: "${DRY_RUN:=false}"
: "${PYTHON_VERSION:=3.11}"
: "${USE_UV:=true}"

# Track-B defaults (can be overridden via environment)
: "${TRACK_B_STEPS:=20000}"
: "${TRACK_B_SEEDS:=0,1,2}"

###############################################################################
# Argument Parsing
###############################################################################
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-setup)
                SKIP_SETUP=true
                shift
                ;;
            --skip-data)
                SKIP_DATA=true
                shift
                ;;
            --quick|-q)
                QUICK_MODE=true
                shift
                ;;
            --dry-run|-d)
                DRY_RUN=true
                shift
                ;;
            --repo-url)
                REPO_URL="$2"
                shift 2
                ;;
            --branch)
                BRANCH="$2"
                shift 2
                ;;
            --work-dir)
                WORK_DIR="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                warn "Unknown argument: $1"
                shift
                ;;
        esac
    done
}

show_help() {
    cat <<EOF
vast.ai Auto Deploy Script for ECD (Embedding Compression DB)

Usage: bash scripts/vastai_deploy.sh [OPTIONS]

Options:
  --skip-setup      Skip environment setup (assume already configured)
  --skip-data       Skip data preparation step  
  --quick, -q       Run quick mode (10 steps instead of full training)
  --dry-run, -d     Show what would run without executing
  --repo-url URL    Git repo URL to clone (if not already in project dir)
  --branch BRANCH   Git branch to checkout (default: main)
  --work-dir DIR    Working directory (default: /workspace/ecd)
  --help, -h        Show this help message

Environment Variables:
  TRACK_B_STEPS     Training steps (default: 20000)
  TRACK_B_SEEDS     Seeds to run (default: 0,1,2)
  SKIP_SETUP        Skip setup if "true"
  SKIP_DATA         Skip data prep if "true"
  QUICK_MODE        Quick mode if "true"

Example vast.ai onstart script:
  #!/bin/bash
  cd /workspace
  git clone https://github.com/your-user/ecd.git
  cd ecd
  bash scripts/vastai_deploy.sh --quick

Example with environment overrides:
  TRACK_B_STEPS=50000 TRACK_B_SEEDS=0 bash scripts/vastai_deploy.sh
EOF
}

###############################################################################
# System Detection
###############################################################################
detect_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        log "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
        return 0
    else
        warn "No NVIDIA GPU detected (nvidia-smi not found)"
        return 1
    fi
}

detect_cuda_version() {
    if command -v nvcc &>/dev/null; then
        nvcc --version | grep "release" | awk '{print $5}' | tr -d ','
    elif [[ -f /usr/local/cuda/version.txt ]]; then
        cat /usr/local/cuda/version.txt | awk '{print $3}'
    else
        echo "unknown"
    fi
}

get_system_info() {
    log "=== System Information ==="
    echo "  Hostname: $(hostname)"
    echo "  OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || uname -a)"
    echo "  CPU Cores: $(nproc 2>/dev/null || echo 'unknown')"
    echo "  Memory: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo 'unknown')"
    echo "  CUDA Version: $(detect_cuda_version)"
    detect_gpu || true
    echo ""
}

###############################################################################
# Environment Setup
###############################################################################
setup_apt_packages() {
    log "Installing system packages..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would install: git curl wget build-essential"
        return 0
    fi
    
    export DEBIAN_FRONTEND=noninteractive
    
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        git \
        curl \
        wget \
        build-essential \
        software-properties-common \
        ca-certificates \
        gnupg \
        lsb-release \
        htop \
        tmux \
        vim \
        || warn "Some apt packages may have failed to install"
}

setup_python() {
    log "Setting up Python ${PYTHON_VERSION}..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would setup Python ${PYTHON_VERSION}"
        return 0
    fi
    
    # Check if Python is already available
    if command -v python3.11 &>/dev/null; then
        log "Python 3.11 already installed"
        PYTHON_CMD="python3.11"
    elif command -v python3 &>/dev/null; then
        local py_version
        py_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ "$(echo "$py_version >= 3.11" | bc -l 2>/dev/null || echo 0)" == "1" ]] || [[ "$py_version" == "3.11" ]] || [[ "$py_version" > "3.10" ]]; then
            log "Python $py_version found (meets >=3.11 requirement)"
            PYTHON_CMD="python3"
        else
            warn "Python $py_version found but need >=3.11, attempting install..."
            install_python_311
        fi
    else
        install_python_311
    fi
}

install_python_311() {
    log "Installing Python 3.11..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev python3-pip
    PYTHON_CMD="python3.11"
}

setup_uv() {
    log "Setting up uv package manager..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would install uv"
        return 0
    fi
    
    if command -v uv &>/dev/null; then
        log "uv already installed: $(uv --version)"
        return 0
    fi
    
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    if command -v uv &>/dev/null; then
        log "uv installed successfully: $(uv --version)"
    else
        warn "uv installation may have failed, falling back to pip"
        USE_UV=false
    fi
}

setup_project_deps() {
    log "Installing project dependencies..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would install Python dependencies"
        return 0
    fi
    
    cd "$WORK_DIR"
    
    if [[ "$USE_UV" == "true" ]] && command -v uv &>/dev/null; then
        log "Using uv to sync dependencies..."
        
        # Create venv if needed
        if [[ ! -d ".venv" ]]; then
            uv venv --python "${PYTHON_CMD:-python3}"
        fi
        
        # Sync from lock file
        if [[ -f "uv.lock" ]]; then
            uv sync
        else
            uv pip install -e ".[dev]"
        fi
        
        # Activate for subsequent commands
        source .venv/bin/activate
    else
        log "Using pip to install dependencies..."
        
        ${PYTHON_CMD:-python3} -m venv .venv
        source .venv/bin/activate
        
        pip install --upgrade pip wheel setuptools
        pip install -e ".[dev]"
    fi
    
    # Install PyTorch with CUDA if available
    if detect_gpu; then
        log "Installing PyTorch with CUDA support..."
        pip install torch --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
        pip install torch --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
        log "Using default PyTorch (CUDA version from pyproject.toml)"
    fi
}

###############################################################################
# Project Setup
###############################################################################
clone_or_update_repo() {
    if [[ -n "$REPO_URL" ]]; then
        log "Cloning repository from $REPO_URL..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            info "[DRY RUN] Would clone $REPO_URL to $WORK_DIR"
            return 0
        fi
        
        mkdir -p "$(dirname "$WORK_DIR")"
        
        if [[ -d "$WORK_DIR/.git" ]]; then
            log "Repository already exists, pulling latest..."
            cd "$WORK_DIR"
            git fetch origin
            git checkout "$BRANCH"
            git pull origin "$BRANCH"
        else
            git clone --branch "$BRANCH" "$REPO_URL" "$WORK_DIR"
            cd "$WORK_DIR"
        fi
    else
        # Assume we're already in the project directory
        if [[ -f "pyproject.toml" ]] && grep -q "embedding-compression-db" pyproject.toml 2>/dev/null; then
            WORK_DIR="$(pwd)"
            log "Using current directory as project root: $WORK_DIR"
        elif [[ -d "$WORK_DIR" ]]; then
            cd "$WORK_DIR"
        else
            die "No REPO_URL provided and not in project directory. Use --repo-url or run from project root."
        fi
    fi
}

###############################################################################
# Data Preparation
###############################################################################
prepare_data() {
    log "Preparing data and caching teacher embeddings..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would run: python scripts/prepare_data.py --run-id vast_deploy"
        return 0
    fi
    
    cd "$WORK_DIR"
    
    # Activate venv if exists
    [[ -f ".venv/bin/activate" ]] && source .venv/bin/activate
    
    # Run data preparation
    python scripts/prepare_data.py --run-id vast_deploy 2>&1 | tee logs/prepare_data.log
}

###############################################################################
# Docker Services (Optional)
###############################################################################
setup_docker_services() {
    log "Setting up Docker services (Qdrant, PostgreSQL)..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would start docker-compose services"
        return 0
    fi
    
    cd "$WORK_DIR"
    
    if ! command -v docker &>/dev/null; then
        warn "Docker not installed, skipping services setup"
        warn "Install Docker or run services externally if needed"
        return 0
    fi
    
    if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
        warn "docker-compose not found, skipping services"
        return 0
    fi
    
    # Start services in background
    if docker compose version &>/dev/null; then
        docker compose up -d
    else
        docker-compose up -d
    fi
    
    log "Waiting for services to be ready..."
    sleep 5
    
    # Health check
    if curl -s http://localhost:6333/health &>/dev/null; then
        log "Qdrant is healthy"
    else
        warn "Qdrant may not be ready yet"
    fi
}

###############################################################################
# Main Training Execution
###############################################################################
run_track_b() {
    log "=========================================="
    log "Starting Track-B Experiments"
    log "=========================================="
    
    cd "$WORK_DIR"
    
    # Activate venv
    [[ -f ".venv/bin/activate" ]] && source .venv/bin/activate
    
    # Build command arguments
    local args=()
    
    if [[ "$QUICK_MODE" == "true" ]]; then
        args+=("--quick")
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        args+=("--dry-run")
    fi
    
    # Create logs directory
    mkdir -p logs runs
    
    # Set environment variables for Track-B
    export STEPS="${TRACK_B_STEPS}"
    export SEEDS="${TRACK_B_SEEDS}"
    
    log "Configuration:"
    log "  Steps: $STEPS"
    log "  Seeds: $SEEDS"
    log "  Quick Mode: $QUICK_MODE"
    log "  Dry Run: $DRY_RUN"
    log ""
    
    # Run Track-B script
    local start_time
    start_time=$(date +%s)
    
    bash scripts/run_track_b.sh "${args[@]}" 2>&1 | tee "logs/track_b_$(date +%Y%m%d_%H%M%S).log"
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log ""
    log "=========================================="
    log "Track-B Experiments Completed"
    log "Duration: $((duration / 3600))h $((duration % 3600 / 60))m $((duration % 60))s"
    log "=========================================="
    
    # Show results summary if available
    if [[ -f "summary.txt" ]]; then
        log ""
        log "Results Summary:"
        cat summary.txt
    fi
}

###############################################################################
# Cleanup & Monitoring
###############################################################################
setup_monitoring() {
    log "Setting up monitoring..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    # Create a simple monitoring script
    cat > "$WORK_DIR/scripts/monitor.sh" << 'MONITOR_EOF'
#!/bin/bash
# Simple GPU monitoring script
while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null || echo "No GPU"
    echo ""
    echo "=== Recent Logs ==="
    tail -20 logs/*.log 2>/dev/null | head -30 || echo "No logs yet"
    sleep 10
done
MONITOR_EOF
    chmod +x "$WORK_DIR/scripts/monitor.sh"
}

create_tmux_session() {
    if ! command -v tmux &>/dev/null; then
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    # Create tmux session for monitoring
    tmux new-session -d -s ecd_monitor 2>/dev/null || true
    tmux send-keys -t ecd_monitor "cd $WORK_DIR && watch -n 5 'nvidia-smi 2>/dev/null; echo; tail -10 logs/*.log 2>/dev/null'" Enter 2>/dev/null || true
    
    log "Monitoring tmux session created. Attach with: tmux attach -t ecd_monitor"
}

###############################################################################
# Main Entry Point
###############################################################################
main() {
    parse_args "$@"
    
    log "=========================================="
    log "vast.ai Auto Deploy - ECD Project"
    log "=========================================="
    log ""
    
    get_system_info
    
    # Step 1: Clone/Update Repository
    clone_or_update_repo
    
    # Step 2: Environment Setup
    if [[ "$SKIP_SETUP" != "true" ]]; then
        log "[1/5] Setting up environment..."
        setup_apt_packages
        setup_python
        setup_uv
        setup_project_deps
        setup_monitoring
    else
        log "[1/5] Skipping setup (--skip-setup)"
        # Still need to activate venv
        [[ -f "$WORK_DIR/.venv/bin/activate" ]] && source "$WORK_DIR/.venv/bin/activate"
    fi
    
    # Step 3: Docker Services (optional)
    log "[2/5] Docker services..."
    setup_docker_services
    
    # Step 4: Data Preparation
    if [[ "$SKIP_DATA" != "true" ]]; then
        log "[3/5] Preparing data..."
        mkdir -p "$WORK_DIR/logs"
        prepare_data
    else
        log "[3/5] Skipping data preparation (--skip-data)"
    fi
    
    # Step 5: Create monitoring
    log "[4/5] Setting up monitoring..."
    create_tmux_session
    
    # Step 6: Run Track-B
    log "[5/5] Running Track-B experiments..."
    run_track_b
    
    log ""
    log "=========================================="
    log "Deployment Complete!"
    log "=========================================="
    log ""
    log "Output files:"
    log "  summary.csv  - Machine-readable results"
    log "  summary.txt  - Human-readable summary"
    log "  runs/        - All experiment runs"
    log "  logs/        - Execution logs"
    log ""
    log "To monitor: tmux attach -t ecd_monitor"
    log ""
}

main "$@"
