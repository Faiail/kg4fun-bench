#!/bin/bash

mkdir -p outs/experiments/ontomap/dataset1/hybrid
echo "ONTOMAP DATASET1 HYBRID"
condor_submit JobBatchName=ontomap_rag ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset1/hybrid/rag.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset1/hybrid/rag.out" ERR="outs/experiments/ontomap/dataset1/hybrid/rag.out" LOG="outs/experiments/ontomap/dataset1/hybrid/rag.log"

condor_submit JobBatchName=ontomap_naiv_conv_oaei ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset1/hybrid/naiv_conv_oaei.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset1/hybrid/naiv_conv_oaei.out" ERR="outs/experiments/ontomap/dataset1/hybrid/naiv_conv_oaei.out" LOG="outs/experiments/ontomap/dataset1/hybrid/naiv_conv_oaei.log"

condor_submit JobBatchName=ontomap_icv ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset1/hybrid/icv.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset1/hybrid/icv.out" ERR="outs/experiments/ontomap/dataset1/hybrid/icv.out" LOG="outs/experiments/ontomap/dataset1/hybrid/icv.log"

condor_submit JobBatchName=ontomap_retrieval ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset1/hybrid/retrieval.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset1/hybrid/retrieval.out" ERR="outs/experiments/ontomap/dataset1/hybrid/retrieval.out" LOG="outs/experiments/ontomap/dataset1/hybrid/retrieval.log"

condor_submit JobBatchName=ontomap_fewshot ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset1/hybrid/fewshot.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset1/hybrid/fewshot.out" ERR="outs/experiments/ontomap/dataset1/hybrid/fewshot.out" LOG="outs/experiments/ontomap/dataset1/hybrid/fewshot.log"

condor_submit JobBatchName=ontomap_lightweight ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset1/hybrid/lightweight.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset1/hybrid/lightweight.out" ERR="outs/experiments/ontomap/dataset1/hybrid/lightweight.out" LOG="outs/experiments/ontomap/dataset1/hybrid/lightweight.log"
