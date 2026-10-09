#!/bin/bash

mkdir -p outs/experiments/text/complete/dataset2/inductive
echo "TEXT COMPLETE DATASET2 INDUCTIVE"
condor_submit JobBatchName=complete_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/shared_bert.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/bert.out" ERR="outs/experiments/text/complete/dataset2/inductive/bert.out" LOG="outs/experiments/text/complete/dataset2/inductive/bert.log"

condor_submit JobBatchName=complete_dedicated_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/dedicated_bert.yaml --cls DedicatedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/dedicated_bert.out" ERR="outs/experiments/text/complete/dataset2/inductive/dedicated_bert.out" LOG="outs/experiments/text/complete/dataset2/inductive/dedicated_bert.log"

condor_submit JobBatchName=complete_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/shared_bge.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/bge.out" ERR="outs/experiments/text/complete/dataset2/inductive/bge.out" LOG="outs/experiments/text/complete/dataset2/inductive/bge.log"

condor_submit JobBatchName=complete_dedicated_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/dedicated_bge.yaml --cls DedicatedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/dedicated_bge.out" ERR="outs/experiments/text/complete/dataset2/inductive/dedicated_bge.out" LOG="outs/experiments/text/complete/dataset2/inductive/dedicated_bge.log"

condor_submit JobBatchName=complete_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/shared_modernbert.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/modernbert.out" ERR="outs/experiments/text/complete/dataset2/inductive/modernbert.out" LOG="outs/experiments/text/complete/dataset2/inductive/modernbert.log"

condor_submit JobBatchName=complete_dedicated_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/dedicated_modernbert.yaml --cls DedicatedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/dedicated_modernbert.out" ERR="outs/experiments/text/complete/dataset2/inductive/dedicated_modernbert.out" LOG="outs/experiments/text/complete/dataset2/inductive/dedicated_modernbert.log"

condor_submit JobBatchName=complete_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/shared_roberta.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/roberta.out" ERR="outs/experiments/text/complete/dataset2/inductive/roberta.out" LOG="outs/experiments/text/complete/dataset2/inductive/roberta.log"

condor_submit JobBatchName=complete_dedicated_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset2/inductive/dedicated_roberta.yaml --cls DedicatedTextCompleteRun" OUT="outs/experiments/text/complete/dataset2/inductive/dedicated_roberta.out" ERR="outs/experiments/text/complete/dataset2/inductive/dedicated_roberta.out" LOG="outs/experiments/text/complete/dataset2/inductive/dedicated_roberta.log"

