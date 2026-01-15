#!/usr/bin/env bash
# Simple progress bar implementation for bash

# Initialize progress bar
# Usage: progress_start <total> <description>
progress_start() {
  local total="$1"
  local desc="$2"

  export PROGRESS_TOTAL="$total"
  export PROGRESS_CURRENT=0
  export PROGRESS_DESC="$desc"

  # Check if terminal supports progress bar
  if [[ -t 1 ]]; then
    export PROGRESS_USE_BAR="true"
  else
    export PROGRESS_USE_BAR="false"
  fi

  progress_update 0
}

# Update progress bar
# Usage: progress_update <increment> [new_description]
progress_update() {
  local increment="$1"
  local new_desc="${2:-}"

  PROGRESS_CURRENT=$((PROGRESS_CURRENT + increment))

  if [[ -n "$new_desc" ]]; then
    PROGRESS_DESC="$new_desc"
  fi

  if [[ "$PROGRESS_USE_BAR" == "true" ]]; then
    local percent=$((PROGRESS_CURRENT * 100 / PROGRESS_TOTAL))
    local bar_width=40
    local filled=$((percent * bar_width / 100))
    local empty=$((bar_width - filled))

    local bar=""
    local i
    for ((i=0; i<filled; i++)); do
      bar="${bar}█"
    done
    for ((i=0; i<empty; i++)); do
      bar="${bar}░"
    done

    printf "\r\033[K"  # Clear line
    printf "[%s] %3d%% %s [%s]" "$PROGRESS_DESC" "$percent" "$bar"
  else
    printf "[%s] %d/%d\n" "$PROGRESS_DESC" "$PROGRESS_CURRENT" "$PROGRESS_TOTAL"
  fi
}

# Finish progress bar
progress_finish() {
  if [[ "$PROGRESS_USE_BAR" == "true" ]]; then
    printf "\r\033[K"
    printf "[%s] %d/%d complete!\n" "$PROGRESS_DESC" "$PROGRESS_TOTAL" "$PROGRESS_TOTAL"
  else
    printf "[%s] %d/%d complete!\n" "$PROGRESS_DESC" "$PROGRESS_CURRENT" "$PROGRESS_TOTAL"
  fi

  unset PROGRESS_TOTAL PROGRESS_CURRENT PROGRESS_DESC PROGRESS_USE_BAR
}
