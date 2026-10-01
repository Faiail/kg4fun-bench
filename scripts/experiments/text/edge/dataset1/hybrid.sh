#!/bin/bash

mkdir -p outs/experiments/text/edge/dataset1/hybrid
echo "TEXT EDGE DATASET1 HYBRID"
condor_submit JobBatchName=edge_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/edge/dataset1/hybrid/shared_bert.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/edge/dataset1/hybrid/bert.out" ERR="outs/experiments/text/edge/dataset1/hybrid/bert.out" LOG="outs/experiments/text/edge/dataset1/hybrid/bert.log"

condor_submit JobBatchName=edge_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/edge/dataset1/hybrid/shared_bge.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/edge/dataset1/hybrid/bge.out" ERR="outs/experiments/text/edge/dataset1/hybrid/bge.out" LOG="outs/experiments/text/edge/dataset1/hybrid/bge.log"

condor_submit JobBatchName=edge_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/edge/dataset1/hybrid/shared_modernbert.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/edge/dataset1/hybrid/modernbert.out" ERR="outs/experiments/text/edge/dataset1/hybrid/modernbert.out" LOG="outs/experiments/text/edge/dataset1/hybrid/modernbert.log"

condor_submit JobBatchName=edge_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/edge/dataset1/hybrid/shared_roberta.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/edge/dataset1/hybrid/roberta.out" ERR="outs/experiments/text/edge/dataset1/hybrid/roberta.out" LOG="outs/experiments/text/edge/dataset1/hybrid/roberta.log"