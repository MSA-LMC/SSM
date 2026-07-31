#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/disfa_dfew_checkpoint.pth [evaluate options]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift
# Resolve the checkpoint before changing to the repository directory.
if [[ "$CHECKPOINT" != /* ]]; then
  CHECKPOINT="$PWD/$CHECKPOINT"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "See README.md for model download links." >&2
  exit 2
fi

# Forward fold and dataset-root overrides to the unified evaluator.
python evaluate.py \
  --config configs/disfa_dfew.json \
  --checkpoint "$CHECKPOINT" \
  "$@"
