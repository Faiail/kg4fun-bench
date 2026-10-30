#!/bin/bash

mkdir -p outs/experiments/text/complete/dataset1/inductive
echo "TEXT COMPLETE DATASET1 INDUCTIVE"
condor_submit JobBatchName=complete_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/inductive/shared_bert.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/inductive/bert.out" ERR="outs/experiments/text/complete/dataset1/inductive/bert.out" LOG="outs/experiments/text/complete/dataset1/inductive/bert.log"

condor_submit JobBatchName=complete_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/inductive/shared_bge.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/inductive/bge.out" ERR="outs/experiments/text/complete/dataset1/inductive/bge.out" LOG="outs/experiments/text/complete/dataset1/inductive/bge.log"

condor_submit JobBatchName=complete_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/inductive/shared_modernbert.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/inductive/modernbert.out" ERR="outs/experiments/text/complete/dataset1/inductive/modernbert.out" LOG="outs/experiments/text/complete/dataset1/inductive/modernbert.log"

condor_submit JobBatchName=complete_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/inductive/shared_roberta.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/inductive/roberta.out" ERR="outs/experiments/text/complete/dataset1/inductive/roberta.out" LOG="outs/experiments/text/complete/dataset1/inductive/roberta.log"