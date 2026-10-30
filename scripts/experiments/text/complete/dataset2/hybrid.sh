#!/bin/bash

mkdir -p outs/experiments/text/complete/dataset2/hybrid
echo "TEXT COMPLETE DATASET2 HYBRID"
condor_submit JobBatchName=complete_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/hybrid/shared_bert.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/hybrid/bert.out" ERR="outs/experiments/text/complete/dataset2/hybrid/bert.out" LOG="outs/experiments/text/complete/dataset2/hybrid/bert.log"

condor_submit JobBatchName=complete_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/hybrid/shared_bge.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/hybrid/bge.out" ERR="outs/experiments/text/complete/dataset2/hybrid/bge.out" LOG="outs/experiments/text/complete/dataset2/hybrid/bge.log"

condor_submit JobBatchName=complete_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/hybrid/shared_modernbert.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/hybrid/modernbert.out" ERR="outs/experiments/text/complete/dataset2/hybrid/modernbert.out" LOG="outs/experiments/text/complete/dataset2/hybrid/modernbert.log"

condor_submit JobBatchName=complete_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/hybrid/shared_roberta.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/hybrid/roberta.out" ERR="outs/experiments/text/complete/dataset2/hybrid/roberta.out" LOG="outs/experiments/text/complete/dataset2/hybrid/roberta.log"