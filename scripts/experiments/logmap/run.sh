#!/bin/bash


uv run main.py experiment --parameters configs/experiment/logmap/dataset1/hybrid.yaml --cls LogmapRun --seed 42
uv run main.py experiment --parameters configs/experiment/logmap/dataset1/inductive.yaml --cls LogmapRun --seed 42

uv run main.py experiment --parameters configs/experiment/logmap/dataset2/hybrid.yaml --cls LogmapRun --seed 42
uv run main.py experiment --parameters configs/experiment/logmap/dataset2/inductive.yaml --cls LogmapRun --seed 42