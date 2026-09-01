# !/bin/bash

echo "Split dataset 1"
uv run  main.py preprocess --parameters configs/preprocess/splits/dataset1.yaml --cls InductiveSplitProcessor

echo "Split dataset 2"
uv run  main.py preprocess --parameters configs/preprocess/splits/dataset2.yaml --cls InductiveSplitProcessor