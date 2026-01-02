#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run --directory "${repo_dir}" python "${repo_dir}/scripts/paper_figs.py" "$@"
