#!/usr/bin/env bash
# ==============================================================================
# Start an interactive GPU session on HTCondor HPC
# ==============================================================================
set -euo pipefail

INTERACTIVE_CONFIG="condor/interactive"

if [[ ! -f "$INTERACTIVE_CONFIG" ]]; then
  echo "Error: $INTERACTIVE_CONFIG not found in $(pwd)" >&2
  exit 1
fi

echo "=================================================================="
echo "  Requesting Interactive GPU Session on HTCondor..."
echo "  CPUs: 8 | RAM: 128GB | GPUs: 1"
echo "  Once scheduled, you will be placed directly into an interactive shell."
echo "=================================================================="

condor_submit -i "$INTERACTIVE_CONFIG"

