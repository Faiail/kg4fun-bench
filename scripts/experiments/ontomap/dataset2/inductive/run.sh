#!/bin/bash

mkdir -p outs/experiments/ontomap/dataset2/inductive
echo "ONTOMAP DATASET2 INDUCTIVE"
condor_submit JobBatchName=ontomap_rag ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset2/inductive/rag.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset2/inductive/rag.out" ERR="outs/experiments/ontomap/dataset2/inductive/rag.out" LOG="outs/experiments/ontomap/dataset2/inductive/rag.log"

condor_submit JobBatchName=ontomap_naiv_conv_oaei ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset2/inductive/naiv_conv_oaei.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset2/inductive/naiv_conv_oaei.out" ERR="outs/experiments/ontomap/dataset2/inductive/naiv_conv_oaei.out" LOG="outs/experiments/ontomap/dataset2/inductive/naiv_conv_oaei.log"

condor_submit JobBatchName=ontomap_icv ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset2/inductive/icv.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset2/inductive/icv.out" ERR="outs/experiments/ontomap/dataset2/inductive/icv.out" LOG="outs/experiments/ontomap/dataset2/inductive/icv.log"

condor_submit JobBatchName=ontomap_retrieval ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset2/inductive/retrieval.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset2/inductive/retrieval.out" ERR="outs/experiments/ontomap/dataset2/inductive/retrieval.out" LOG="outs/experiments/ontomap/dataset2/inductive/retrieval.log"

condor_submit JobBatchName=ontomap_fewshot ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset2/inductive/fewshot.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset2/inductive/fewshot.out" ERR="outs/experiments/ontomap/dataset2/inductive/fewshot.out" LOG="outs/experiments/ontomap/dataset2/inductive/fewshot.log"

condor_submit JobBatchName=ontomap_lightweight ./condor/launch ARGS="experiment --parameters configs/experiment/ontomap/dataset2/inductive/lightweight.yaml --cls OntomapRun --seed 42" OUT="outs/experiments/ontomap/dataset2/inductive/lightweight.out" ERR="outs/experiments/ontomap/dataset2/inductive/lightweight.out" LOG="outs/experiments/ontomap/dataset2/inductive/lightweight.log"
