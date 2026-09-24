#!/bin/bash

echo "Running MELT preprocessing for dataset1 - Hybrid Test Split"
uv run main.py preprocess --parameters configs/preprocess/melt/dataset1_hybrid.yaml --cls MeltPreprocessor

echo "Running MELT preprocessing for dataset1 - Inductive Test Split"
uv run main.py preprocess --parameters configs/preprocess/melt/dataset1_inductive.yaml --cls MeltPreprocessor

echo "Running MELT preprocessing for dataset2 - Hybrid Test Split"
uv run main.py preprocess --parameters configs/preprocess/melt/dataset2_hybrid.yaml --cls MeltPreprocessor

echo "Running MELT preprocessing for dataset2 - Inductive Test Split"
uv run main.py preprocess --parameters configs/preprocess/melt/dataset2_inductive.yaml --cls MeltPreprocessor

echo "All MELT XML formatting completed successfully!"
