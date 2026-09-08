#!/bin/bash

set -e

kg4fun experiment --parameters configs/experiment/ontomap/dataset1/hybrid/rag.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/dataset1/hybrid/naiv_conv_oaei.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/dataset1/hybrid/icv.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/dataset1/hybrid/retrieval.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/dataset1/hybrid/fewshot.yaml --cls OntomapRun --seed 42
kg4fun experiment --parameters configs/experiment/ontomap/dataset1/hybrid/lightweight.yaml --cls OntomapRun --seed 42
