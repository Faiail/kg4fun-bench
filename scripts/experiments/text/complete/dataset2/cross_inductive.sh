#!/bin/bash

mkdir -p outs/experiments/text/complete/dataset2/inductive
echo "TEXT CROSS COMPLETE DATASET2 INDUCTIVE"

condor_submit JobBatchName=cross_complete_bert_dataset2_inductive ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/cross_bert.yaml --cls SharedCrossTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/cross_bert.out" ERR="outs/experiments/text/complete/dataset2/inductive/cross_bert.out" LOG="outs/experiments/text/complete/dataset2/inductive/cross_bert.log"

condor_submit JobBatchName=cross_complete_bge_dataset2_inductive ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/cross_bge.yaml --cls SharedCrossTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/cross_bge.out" ERR="outs/experiments/text/complete/dataset2/inductive/cross_bge.out" LOG="outs/experiments/text/complete/dataset2/inductive/cross_bge.log"

condor_submit JobBatchName=cross_complete_modernbert_dataset2_inductive ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/cross_modernbert.yaml --cls SharedCrossTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/cross_modernbert.out" ERR="outs/experiments/text/complete/dataset2/inductive/cross_modernbert.out" LOG="outs/experiments/text/complete/dataset2/inductive/cross_modernbert.log"

condor_submit JobBatchName=cross_complete_roberta_dataset2_inductive ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/cross_roberta.yaml --cls SharedCrossTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/cross_roberta.out" ERR="outs/experiments/text/complete/dataset2/inductive/cross_roberta.out" LOG="outs/experiments/text/complete/dataset2/inductive/cross_roberta.log"
