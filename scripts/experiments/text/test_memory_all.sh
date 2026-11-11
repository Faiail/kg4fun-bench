#!/usr/bin/env bash
# ==============================================================================
# Cross-Encoding Memory Profiling Suite
#
# Runs cross-encoding experiments in worst-case mode (warmup=0, all layers active)
# to measure peak VRAM utilization and safely detect OOMs.
#
# Usage:
#   ./scripts/experiments/text/test_memory_all.sh [options]
#
# Options:
#   -t, --task <node|edge|complete|all>           Target task (default: all)
#   -m, --model <bert|bge|modernbert|roberta|all> Target model (default: all)
#   -d, --dataset <dataset1|dataset2>             Target dataset (default: dataset1)
#   -s, --split <hybrid|inductive>                Target split (default: hybrid)
#   -b, --batch-size <int>                        Override batch size (e.g. 512, 256, 128, 64)
#   -H, --hidden-size <int>                       Override final hidden size (e.g. 128, 96)
#   -n, --steps <int>                             Batches to run per model (default: 10)
#   --full-epoch                                  Run a full epoch instead of fixed steps
#   -h, --help                                    Show this help message
#
# Examples:
#   # Quick test across all 3 tasks with 10 batches each:
#   ./scripts/experiments/text/test_memory_all.sh
#
#   # Full epoch test on Complete task with batch size 64:
#   ./scripts/experiments/text/test_memory_all.sh -t complete -b 64 --full-epoch
# ==============================================================================

set -euo pipefail

TASK_FILTER="all"
MODEL_FILTER="all"
DATASET="dataset1"
SPLIT="hybrid"
BATCH_SIZE=""
HIDDEN_SIZE=""
STEPS=10
FULL_EPOCH=false

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Cross-Encoding VRAM Profiler (runs in an interactive GPU session).

Options:
  -t, --task TASK         node, edge, complete, or all (default: all)
  -m, --model MODEL       bert, bge, modernbert, roberta, or all (default: all)
  -d, --dataset DATASET   dataset1 or dataset2 (default: dataset1)
  -s, --split SPLIT       hybrid or inductive (default: hybrid)
  -b, --batch-size BS     Override batch size (e.g. 512, 256, 128, 64)
  -H, --hidden-size HS    Override final hidden size (e.g. 128, 96)
  -n, --steps N           Number of batches to run per model (default: 10)
  --full-epoch            Run the complete epoch instead of N batches
  -h, --help              Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--task)
      TASK_FILTER="$2"
      shift 2
      ;;
    -m|--model)
      MODEL_FILTER="$2"
      shift 2
      ;;
    -d|--dataset)
      DATASET="$2"
      shift 2
      ;;
    -s|--split)
      SPLIT="$2"
      shift 2
      ;;
    -b|--batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -H|--hidden-size)
      HIDDEN_SIZE="$2"
      shift 2
      ;;
    -n|--steps)
      STEPS="$2"
      shift 2
      ;;
    --full-epoch)
      FULL_EPOCH=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

TASKS=("node" "edge" "complete")
MODELS=("bert" "bge" "modernbert" "roberta")

# Activate virtualenv if present
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

REPORT_JSON="/tmp/profile_report_$$.json"
echo "[]" > "$REPORT_JSON"

echo "=================================================================="
echo "  CROSS-ENCODING VRAM PROFILING SUITE"
echo "  Dataset: $DATASET | Split: $SPLIT"
echo "  Task:    $TASK_FILTER | Model: $MODEL_FILTER"
echo "  Steps:   $([ "$FULL_EPOCH" = true ] && echo "FULL EPOCH" || echo "${STEPS} batches")"
[[ -n "$BATCH_SIZE" ]] && echo "  Custom Batch Size: $BATCH_SIZE"
[[ -n "$HIDDEN_SIZE" ]] && echo "  Custom Hidden Dim: $HIDDEN_SIZE"
echo "=================================================================="

for task in "${TASKS[@]}"; do
  [[ "$TASK_FILTER" != "all" && "$TASK_FILTER" != "$task" ]] && continue

  if [[ "$task" == "complete" ]]; then
    RUN_CLS="SharedCrossTextCompleteRun"
  else
    RUN_CLS="SharedCrossTextSingleItemRun"
  fi

  for model in "${MODELS[@]}"; do
    [[ "$MODEL_FILTER" != "all" && "$MODEL_FILTER" != "$model" ]] && continue

    CONFIG="configs/experiment/text/${task}/${DATASET}/${SPLIT}/cross_${model}.yaml"
    if [[ ! -f "$CONFIG" ]]; then
      echo "Warning: Config file '$CONFIG' not found, skipping."
      continue
    fi

    # Build python CLI command
    CMD_ARGS=(
      "main.py" "profile"
      "--parameters" "$CONFIG"
      "--cls" "$RUN_CLS"
      "--steps" "$STEPS"
    )
    if [[ "$FULL_EPOCH" = true ]]; then
      CMD_ARGS+=("--full-epoch")
    fi
    if [[ -n "$BATCH_SIZE" ]]; then
      CMD_ARGS+=("--batch-size" "$BATCH_SIZE")
    fi
    if [[ -n "$HIDDEN_SIZE" ]]; then
      CMD_ARGS+=("--hidden-size" "$HIDDEN_SIZE")
    fi

    echo ""
    echo ">>> Running test: [${task^^}] - [${model^^}] (${CONFIG}) <<<"

    TMP_OUT=$(mktemp)
    if python "${CMD_ARGS[@]}" > "$TMP_OUT" 2>&1; then
      RUN_EXIT=0
    else
      RUN_EXIT=$?
    fi

    # Print log output
    cat "$TMP_OUT"

    # Extract PROFILE_RESULT line
    RESULT_LINE=$(grep "^PROFILE_RESULT:" "$TMP_OUT" | tail -n 1 | sed 's/^PROFILE_RESULT: //')

    if [[ -n "$RESULT_LINE" ]]; then
      python - "$RESULT_LINE" "$REPORT_JSON" "$task" "$model" << 'EOF'
import json, sys

result_line = sys.argv[1]
report_file = sys.argv[2]
task_name = sys.argv[3]
model_name = sys.argv[4]

data = json.loads(result_line)
data["task"] = task_name
data["model"] = model_name

with open(report_file, "r") as f:
    reports = json.load(f)

reports.append(data)

with open(report_file, "w") as f:
    json.dump(reports, f, indent=2)
EOF
    else
      echo "[Error] Could not find PROFILE_RESULT for ${task}_${model} (Exit code: $RUN_EXIT)"
      python - "$REPORT_JSON" "$task" "$model" "${BATCH_SIZE:-default}" "${HIDDEN_SIZE:-default}" << 'EOF'
import json, sys
report_file = sys.argv[1]
task_name = sys.argv[2]
model_name = sys.argv[3]
bs_val = sys.argv[4]
hs_val = sys.argv[5]

with open(report_file, "r") as f:
    reports = json.load(f)

reports.append({
    "task": task_name,
    "model": model_name,
    "batch_size": bs_val,
    "hidden_size": hs_val,
    "status": "FAILED",
    "peak_allocated_gb": 0.0,
    "peak_reserved_gb": 0.0,
    "total_vram_gb": 0.0,
    "pct_used": 0.0,
    "error_message": "Process exited without reporting metrics"
})

with open(report_file, "w") as f:
    json.dump(reports, f, indent=2)
EOF
    fi

    rm -f "$TMP_OUT"
  done
done

# Render Final Formatted Table
echo ""
echo "=========================================================================================="
echo "                                MEMORY PROFILING REPORT"
echo "=========================================================================================="

python - "$REPORT_JSON" << 'EOF'
import json, sys

report_file = sys.argv[1]
with open(report_file) as f:
    reports = json.load(f)

if not reports:
    print("No experiments were profiled.")
    sys.exit(0)

headers = ["Task", "Model", "Batch Size", "Hidden Dim", "Steps", "Allocated", "Reserved", "VRAM Util", "Status"]
rows = []

for r in reports:
    status = r.get("status", "UNKNOWN")
    if status == "PASS":
        pct = r.get("pct_used", 0)
        if pct > 85.0:
            status_str = "WARNING (>85%)"
        else:
            status_str = "PASS"
    elif status == "OOM":
        status_str = "OOM"
    else:
        status_str = status

    steps_str = f"{r.get('steps_completed', 0)}/{r.get('steps_target', 0)}"
    alloc_str = f"{r.get('peak_allocated_gb', 0):.1f} GB"
    resv_str = f"{r.get('peak_reserved_gb', 0):.1f} / {r.get('total_vram_gb', 0):.1f} GB"
    pct_str = f"{r.get('pct_used', 0):.1f}%"

    rows.append([
        r.get("task", "").upper(),
        r.get("model", ""),
        str(r.get("batch_size", "")),
        str(r.get("hidden_size", "")),
        steps_str,
        alloc_str,
        resv_str,
        pct_str,
        status_str
    ])

# Compute column widths
col_widths = [len(h) for h in headers]
for row in rows:
    for idx, cell in enumerate(row):
        col_widths[idx] = max(col_widths[idx], len(cell))

fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
sep = "-+-".join(["-" * w for w in col_widths])

print(fmt.format(*headers))
print(sep)
for row in rows:
    print(fmt.format(*row))

print("==========================================================================================")

has_oom = any(r.get("status") == "OOM" for r in reports)
high_vram = any(r.get("pct_used", 0) > 85.0 for r in reports)

if has_oom:
    print("\n[RECOMMENDATION]: OOM detected in one or more configurations.")
    print("  -> For COMPLETE or EDGE: Halve the batch size (-b 256, -b 128, or -b 64).")
    print("  -> Because attention scales quadratically O(B^2), cutting batch size drops memory by 4x-16x.")
elif high_vram:
    print("\n[RECOMMENDATION]: Some models exceed 85% VRAM utilization.")
    print("  -> Consider dropping batch size by 2x to avoid spikes on batches with longer text sequences.")
else:
    print("\n[VERDICT]: All tested configurations fit comfortably within available VRAM! Safe to launch.")
EOF

# Save report
mkdir -p outs
cp "$REPORT_JSON" outs/memory_profile_report.json
rm -f "$REPORT_JSON"
echo ""
echo "Full JSON report saved to: outs/memory_profile_report.json"
