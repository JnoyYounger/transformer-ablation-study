#!/usr/bin/env bash
set -euo pipefail

# Determine repository root (parent of scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer local venv python if available; allow override via $PYTHON_BIN
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

usage() {
  cat <<'EOF'
Usage: scripts/run.sh <main|ablation> [args...]

Examples:
  scripts/run.sh main --train_path data/train.csv --val_path data/val.csv --save_training_log --save_plot
  scripts/run.sh ablation --only relative_pe_only
  scripts/run.sh ablation --use_full_data

Notes:
  - Extra flags after the subcommand are forwarded to the Python script.
  - If .venv exists, its Python will be used automatically.
  - Set PYTHON_BIN to override the Python interpreter.
EOF
}

if [[ "${1:-}" == "" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CMD="$1"
shift || true

case "$CMD" in
  main)
    exec "$PYTHON_BIN" "$ROOT_DIR/main.py" "$@"
    ;;
  ablation|run_ablation)
    exec "$PYTHON_BIN" "$ROOT_DIR/run_ablation.py" "$@"
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    usage
    exit 1
    ;;
esac