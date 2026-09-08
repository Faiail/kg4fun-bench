#!/bin/bash

set -e

kg4fun experiment --parameters configs/experiment/ontomap/dataset2/inductive/rag.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/dataset2/inductive/naiv_conv_oaei.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/dataset2/inductive/icv.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/dataset2/inductive/retrieval.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/dataset2/inductive/fewshot.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/dataset2/inductive/lightweight.yaml --cls OntomapRun --seed 42
