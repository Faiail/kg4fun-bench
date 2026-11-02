#!/usr/bin/env bash
# ==============================================================================
# Launch Cross-Encoding Text Model Experiments on HTCondor HPC
#
# Supports all 48 benchmark cross-encoding experiments:
#   Tasks:    node, edge, complete
#   Datasets: dataset1, dataset2
#   Splits:   hybrid, inductive
#   Models:   bert, bge, modernbert, roberta
#
# Usage:
#   ./scripts/experiments/text/run_cross_experiments.sh [options]
#   ./scripts/experiments/text/run_cross_experiments.sh [task] [dataset] [split] [model]
#
# Options:
#   -t, --task <node|edge|complete|all>           Target task (default: all)
#   -d, --dataset <dataset1|dataset2|all>          Target dataset (default: all)
#   -s, --split <hybrid|inductive|all>             Target split (default: all)
#   -m, --model <bert|bge|modernbert|roberta|all> Target model (default: all)
#   --dry-run                                     Print condor_submit commands without running
#   -h, --help                                    Display this help message
# ==============================================================================

set -euo pipefail

TASK_FILTER="all"
DATASET_FILTER="all"
SPLIT_FILTER="all"
MODEL_FILTER="all"
DRY_RUN=false

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS] [TASK] [DATASET] [SPLIT] [MODEL]

Submit cross-encoding experiments to HTCondor cluster.

Positional arguments (optional):
  TASK        node, edge, complete, or all (default: all)
  DATASET     dataset1, dataset2, or all (default: all)
  SPLIT       hybrid, inductive, or all (default: all)
  MODEL       bert, bge, modernbert, roberta, or all (default: all)

Options:
  -t, --task TASK         Task: node, edge, complete, or all
  -d, --dataset DATASET   Dataset: dataset1, dataset2, or all
  -s, --split SPLIT       Split: hybrid, inductive, or all
  -m, --model MODEL       Model: bert, bge, modernbert, roberta, or all
  --dry-run               Print submit commands without executing
  -h, --help              Show this help message and exit
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--task)
      TASK_FILTER="$2"
      shift 2
      ;;
    -d|--dataset)
      DATASET_FILTER="$2"
      shift 2
      ;;
    -s|--split)
      SPLIT_FILTER="$2"
      shift 2
      ;;
    -m|--model)
      MODEL_FILTER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ge 1 && "$TASK_FILTER" == "all" ]]; then
  TASK_FILTER="${POSITIONAL[0]}"
fi
if [[ ${#POSITIONAL[@]} -ge 2 && "$DATASET_FILTER" == "all" ]]; then
  DATASET_FILTER="${POSITIONAL[1]}"
fi
if [[ ${#POSITIONAL[@]} -ge 3 && "$SPLIT_FILTER" == "all" ]]; then
  SPLIT_FILTER="${POSITIONAL[2]}"
fi
if [[ ${#POSITIONAL[@]} -ge 4 && "$MODEL_FILTER" == "all" ]]; then
  MODEL_FILTER="${POSITIONAL[3]}"
fi

TASKS=("node" "edge" "complete")
DATASETS=("dataset1" "dataset2")
SPLITS=("hybrid" "inductive")
MODELS=("bert" "bge" "modernbert" "roberta")

validate_item() {
  local item="$1"
  shift
  local valid_items=("$@")
  if [[ "$item" == "all" ]]; then
    return 0
  fi
  for v in "${valid_items[@]}"; do
    if [[ "$v" == "$item" ]]; then
      return 0
    fi
  done
  echo "Error: Invalid value '$item'. Expected one of: ${valid_items[*]} all" >&2
  exit 1
}

validate_item "$TASK_FILTER" "${TASKS[@]}"
validate_item "$DATASET_FILTER" "${DATASETS[@]}"
validate_item "$SPLIT_FILTER" "${SPLITS[@]}"
validate_item "$MODEL_FILTER" "${MODELS[@]}"

CONDOR_LAUNCH="./condor/launch"
if [[ ! -f "$CONDOR_LAUNCH" ]]; then
  echo "Error: HTCondor launch file '$CONDOR_LAUNCH' not found in $(pwd)." >&2
  exit 1
fi

COUNT=0
TOTAL_CONFIGS=0

echo "=================================================================="
echo "  CROSS-ENCODING HPC SUBMIT LAUNCHER"
echo "  Mode:    $([ "$DRY_RUN" = true ] && echo "DRY-RUN (no jobs submitted)" || echo "SUBMITTING TO CONDOR")"
echo "  Filters: task=${TASK_FILTER}, dataset=${DATASET_FILTER}, split=${SPLIT_FILTER}, model=${MODEL_FILTER}"
echo "=================================================================="

for task in "${TASKS[@]}"; do
  [[ "$TASK_FILTER" != "all" && "$TASK_FILTER" != "$task" ]] && continue

  if [[ "$task" == "complete" ]]; then
    RUN_CLS="SharedCrossTextCompleteRun"
  else
    RUN_CLS="SharedCrossTextSingleItemRun"
  fi

  for dataset in "${DATASETS[@]}"; do
    [[ "$DATASET_FILTER" != "all" && "$DATASET_FILTER" != "$dataset" ]] && continue

    for split in "${SPLITS[@]}"; do
      [[ "$SPLIT_FILTER" != "all" && "$SPLIT_FILTER" != "$split" ]] && continue

      OUT_DIR="outs/experiments/text/${task}/${dataset}/${split}"
      if [[ "$DRY_RUN" != true ]]; then
        mkdir -p "$OUT_DIR"
      fi

      for model in "${MODELS[@]}"; do
        [[ "$MODEL_FILTER" != "all" && "$MODEL_FILTER" != "$model" ]] && continue

        CONFIG="configs/experiment/text/${task}/${dataset}/${split}/cross_${model}.yaml"
        if [[ ! -f "$CONFIG" ]]; then
          echo "Warning: Config file '$CONFIG' not found! Skipping..." >&2
          continue
        fi

        BATCH_NAME="cross_${task}_${model}_${dataset}_${split}"
        OUT_FILE="${OUT_DIR}/cross_${model}.out"
        LOG_FILE="${OUT_DIR}/cross_${model}.log"

        TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))

        if [[ "$DRY_RUN" == true ]]; then
          echo "[DRY-RUN] condor_submit JobBatchName=${BATCH_NAME} ${CONDOR_LAUNCH} ARGS=\"experiment --parameters=${CONFIG} --cls ${RUN_CLS}\" OUT=\"${OUT_FILE}\" ERR=\"${OUT_FILE}\" LOG=\"${LOG_FILE}\""
        else
          echo "Submitting: ${BATCH_NAME} (${CONFIG})"
          condor_submit JobBatchName="${BATCH_NAME}" "${CONDOR_LAUNCH}" \
            ARGS="experiment --parameters=${CONFIG} --cls ${RUN_CLS}" \
            OUT="${OUT_FILE}" \
            ERR="${OUT_FILE}" \
            LOG="${LOG_FILE}"
          COUNT=$((COUNT + 1))
        fi
      done
    done
  done
done

echo "=================================================================="
if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run complete. Verified ${TOTAL_CONFIGS} matching experiment job(s)."
else
  echo "Submission complete. Successfully submitted ${COUNT} / ${TOTAL_CONFIGS} job(s)."
fi
echo "=================================================================="

