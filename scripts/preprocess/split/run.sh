# !/bin/bash

echo "Split dataset 1 - Inductive"
uv run main.py preprocess --parameters configs/preprocess/splits/inductive/dataset1.yaml --cls InductiveSplitProcessor

echo "Split dataset 2 - Inductive"
uv run main.py preprocess --parameters configs/preprocess/splits/inductive/dataset2.yaml --cls InductiveSplitProcessor

echo "Split dataset 1 - Hybrid"
uv run main.py preprocess --parameters configs/preprocess/splits/hybrid/dataset1.yaml --cls HybridSplitProcessor

echo "Split dataset 2 - Hybrid"
uv run main.py preprocess --parameters configs/preprocess/splits/hybrid/dataset2.yaml --cls HybridSplitProcessor