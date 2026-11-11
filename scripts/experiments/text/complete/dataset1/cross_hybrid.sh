#!/bin/bash

mkdir -p outs/experiments/text/complete/dataset1/hybrid
echo "TEXT CROSS COMPLETE DATASET1 HYBRID"

condor_submit JobBatchName=cross_complete_bert_dataset1_hybrid ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/cross_bert.yaml --cls SharedCrossTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/cross_bert.out" ERR="outs/experiments/text/complete/dataset1/hybrid/cross_bert.out" LOG="outs/experiments/text/complete/dataset1/hybrid/cross_bert.log"

condor_submit JobBatchName=cross_complete_bge_dataset1_hybrid ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/cross_bge.yaml --cls SharedCrossTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/cross_bge.out" ERR="outs/experiments/text/complete/dataset1/hybrid/cross_bge.out" LOG="outs/experiments/text/complete/dataset1/hybrid/cross_bge.log"

condor_submit JobBatchName=cross_complete_modernbert_dataset1_hybrid ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/cross_modernbert.yaml --cls SharedCrossTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/cross_modernbert.out" ERR="outs/experiments/text/complete/dataset1/hybrid/cross_modernbert.out" LOG="outs/experiments/text/complete/dataset1/hybrid/cross_modernbert.log"

condor_submit JobBatchName=cross_complete_roberta_dataset1_hybrid ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/cross_roberta.yaml --cls SharedCrossTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/cross_roberta.out" ERR="outs/experiments/text/complete/dataset1/hybrid/cross_roberta.out" LOG="outs/experiments/text/complete/dataset1/hybrid/cross_roberta.log"
