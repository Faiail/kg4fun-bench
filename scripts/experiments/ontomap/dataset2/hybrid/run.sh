#!/bin/bash


uv run main.py experiment --parameters configs/experiment/ontomap/dataset2/hybrid/rag.yaml --cls OntomapRun --seed 42
uv run main.py experiment --parameters configs/experiment/ontomap/dataset2/hybrid/naiv_conv_oaei.yaml --cls OntomapRun --seed 42
uv run main.py experiment --parameters configs/experiment/ontomap/dataset2/hybrid/icv.yaml --cls OntomapRun --seed 42
uv run main.py experiment --parameters configs/experiment/ontomap/dataset2/hybrid/retrieval.yaml --cls OntomapRun --seed 42
uv run main.py experiment --parameters configs/experiment/ontomap/dataset2/hybrid/fewshot.yaml --cls OntomapRun --seed 42
uv run main.py experiment --parameters configs/experiment/ontomap/dataset2/hybrid/lightweight.yaml --cls OntomapRun --seed 42
