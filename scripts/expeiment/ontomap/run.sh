#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

kg4fun experiment --parameters configs/experiment/ontomap/fewshot.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/icv.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/lightweight.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/naiv_conv_oaei.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/rag.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/retrieval.yaml --cls OntomapRun --seed 42

