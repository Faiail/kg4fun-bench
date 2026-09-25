#!/bin/bash

echo "Running LOGMAP preprocessing for dataset1 - Hybrid Test Split"
uv run main.py preprocess --parameters configs/preprocess/logmap/dataset1_hybrid.yaml --cls LogMapPreprocessor

echo "Running LOGMAP preprocessing for dataset1 - Inductive Test Split"
uv run main.py preprocess --parameters configs/preprocess/logmap/dataset1_inductive.yaml --cls LogMapPreprocessor

echo "Running LOGMAP preprocessing for dataset2 - Hybrid Test Split"
uv run main.py preprocess --parameters configs/preprocess/logmap/dataset2_hybrid.yaml --cls LogMapPreprocessor

echo "Running LOGMAP preprocessing for dataset2 - Inductive Test Split"
uv run main.py preprocess --parameters configs/preprocess/logmap/dataset2_inductive.yaml --cls LogMapPreprocessor

echo "All LOGMAP XML formatting completed successfully!"
