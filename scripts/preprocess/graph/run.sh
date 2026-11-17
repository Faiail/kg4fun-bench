#!/bin/bash
set -e

# Always run from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASETS=("dataset1" "dataset2")
SETTINGS=("hybrid" "inductive")
SPLITS=("train" "val" "test")

for ds in "${DATASETS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
        for split in "${SPLITS[@]}"; do
            echo "=========================================================="
            echo "Processing ${ds} | ${setting} | ${split}"
            echo "=========================================================="

            schema_cfg="configs/preprocess/graph/schema/${ds}/${setting}/${split}.yaml"
            input_cfg="configs/preprocess/graph/input_kg/${ds}/${setting}/${split}.yaml"

            echo "--> 1. Converting Schema KG..."
            uv run main.py preprocess --parameters "${schema_cfg}" --cls SchemaConverter

            echo "--> 2. Converting Input KG & Generating Annotations..."
            uv run main.py preprocess --parameters "${input_cfg}" --cls InputKGConverter

            echo "Finished ${ds} | ${setting} | ${split} successfully!"
            echo ""
        done
    done
done

echo "All graph conversions completed successfully!"
