#!/usr/bin/env bash
###############################################################################
# vast.ai Onstart Script Template
#
# Copy this into your vast.ai instance's "On-start Script" field, or use as
# a template for your custom deployment.
#
# This is a minimal bootstrap that:
#   1. Clones the repository
#   2. Runs the full deployment script
#
# Customize the variables below for your setup.
###############################################################################
set -euo pipefail

# ============================================================================
# CUSTOMIZE THESE VARIABLES
# ============================================================================
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/ecd.git}"
BRANCH="${BRANCH:-main}"
WORK_DIR="${WORK_DIR:-/workspace/ecd}"

# Track-B experiment settings
export TRACK_B_STEPS="${TRACK_B_STEPS:-20000}"
export TRACK_B_SEEDS="${TRACK_B_SEEDS:-0,1,2}"

# Options: set to "true" to enable
QUICK_MODE="${QUICK_MODE:-false}"
SKIP_DATA="${SKIP_DATA:-false}"

# ============================================================================
# DEPLOYMENT
# ============================================================================

echo "=========================================="
echo "vast.ai Onstart - ECD Project"
echo "=========================================="
echo "Repository: $REPO_URL"
echo "Branch: $BRANCH"
echo "Work Dir: $WORK_DIR"
echo ""

# Install git if not present
if ! command -v git &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq git
fi

# Clone repository
mkdir -p "$(dirname "$WORK_DIR")"
if [[ -d "$WORK_DIR/.git" ]]; then
    echo "Repository exists, pulling latest..."
    cd "$WORK_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "Cloning repository..."
    git clone --branch "$BRANCH" "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

# Build deployment arguments
DEPLOY_ARGS=()
[[ "$QUICK_MODE" == "true" ]] && DEPLOY_ARGS+=("--quick")
[[ "$SKIP_DATA" == "true" ]] && DEPLOY_ARGS+=("--skip-data")

# Run the main deployment script
echo ""
echo "Running deployment script..."
bash scripts/vastai_deploy.sh "${DEPLOY_ARGS[@]}"
