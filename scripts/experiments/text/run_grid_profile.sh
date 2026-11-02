#!/usr/bin/env bash
# ==============================================================================
# Cross-Encoding Memory Grid Search Runner
#
# Executable script invoked by HTCondor or directly in bash.
# ==============================================================================

set -euo pipefail

mkdir -p outs

# Activate virtual environment
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

python scripts/experiments/text/grid_search_memory.py "$@"

