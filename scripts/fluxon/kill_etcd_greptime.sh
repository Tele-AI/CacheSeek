#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="fluxon_kv_stack"


main() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "required command not found: tmux"
    exit 1
  fi

  if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session not found: $SESSION_NAME"
    exit 1
  fi

  tmux kill-session -t "$SESSION_NAME"
  echo "killed tmux session: $SESSION_NAME"
}


main "$@"

