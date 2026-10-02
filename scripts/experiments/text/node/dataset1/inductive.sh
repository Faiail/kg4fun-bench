#!/bin/bash

mkdir -p outs/experiments/text/node/dataset1/inductive
echo "TEXT NODE DATASET1 INDUCTIVE"
# condor_submit JobBatchName=node_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/shared_bert.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/bert.out" ERR="outs/experiments/text/node/dataset1/inductive/bert.out" LOG="outs/experiments/text/node/dataset1/inductive/bert.log"

condor_submit JobBatchName=node_dedicated_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/dedicated_bert.yaml --cls DedicatedTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/dedicated_bert.out" ERR="outs/experiments/text/node/dataset1/inductive/dedicated_bert.out" LOG="outs/experiments/text/node/dataset1/inductive/dedicated_bert.log"

# condor_submit JobBatchName=node_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/shared_bge.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/bge.out" ERR="outs/experiments/text/node/dataset1/inductive/bge.out" LOG="outs/experiments/text/node/dataset1/inductive/bge.log"

condor_submit JobBatchName=node_dedicated_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/dedicated_bge.yaml --cls DedicatedTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/dedicated_bge.out" ERR="outs/experiments/text/node/dataset1/inductive/dedicated_bge.out" LOG="outs/experiments/text/node/dataset1/inductive/dedicated_bge.log"

# condor_submit JobBatchName=node_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/shared_modernbert.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/modernbert.out" ERR="outs/experiments/text/node/dataset1/inductive/modernbert.out" LOG="outs/experiments/text/node/dataset1/inductive/modernbert.log"

condor_submit JobBatchName=node_dedicated_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/dedicated_modernbert.yaml --cls DedicatedTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/dedicated_modernbert.out" ERR="outs/experiments/text/node/dataset1/inductive/dedicated_modernbert.out" LOG="outs/experiments/text/node/dataset1/inductive/dedicated_modernbert.log"

# condor_submit JobBatchName=node_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/shared_roberta.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/roberta.out" ERR="outs/experiments/text/node/dataset1/inductive/roberta.out" LOG="outs/experiments/text/node/dataset1/inductive/roberta.log"

condor_submit JobBatchName=node_dedicated_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/dedicated_roberta.yaml --cls DedicatedTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/dedicated_roberta.out" ERR="outs/experiments/text/node/dataset1/inductive/dedicated_roberta.out" LOG="outs/experiments/text/node/dataset1/inductive/dedicated_roberta.log"