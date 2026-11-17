#!/bin/bash
set -e

# Always run from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${1:-bge}"

echo "=========================================================="
echo "Starting raw node feature extraction using model: ${MODEL}"
echo "=========================================================="

DATASETS=("dataset1" "dataset2")
SETTINGS=("hybrid" "inductive")
SPLITS=("train" "val" "test")

TARGET_ENTITIES=(
    "schema:Class"
    "schema:Relation"
    "input_kg:Node"
    "input_kg:Edge"
    "input_kg:MetaEdge"
)

for ds in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
        for split in "${SPLITS[@]}"; do
            echo "=========================================================="
            echo "Processing ${ds} | ${setting} | ${split} [${MODEL}]"
            echo "=========================================================="

            for item in "${TARGET_ENTITIES[@]}"; do
                target="${item%%:*}"
                node_t="${item##*:}"
                cfg="configs/preprocess/graph/features/${MODEL}/${ds}/${setting}/${split}/${target}/${node_t}.yaml"
                if [ ! -f "${cfg}" ]; then
                    echo "Config file not found: ${cfg}, skipping..."
                    continue
                fi

                echo "--> Extracting features for [${target}] ${node_t}..."
                uv run main.py preprocess --parameters "${cfg}" --cls NodeFeatureExtractor
            done

            echo "Finished ${ds} | ${setting} | ${split} successfully!"
            echo ""
        done
    done
done

echo "All raw node feature extractions for ${MODEL} completed successfully!"
