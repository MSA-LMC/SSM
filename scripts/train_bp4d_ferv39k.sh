#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root while forwarding optional training arguments.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python train.py --config configs/bp4d_ferv39k.json "$@"
