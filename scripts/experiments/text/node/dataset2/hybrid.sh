#!/bin/bash

mkdir -p outs/experiments/text/node/dataset2/hybrid
echo "TEXT NODE DATASET2 HYBRID"
condor_submit JobBatchName=node_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/hybrid/shared_bert.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/dataset2/hybrid/bert.out" ERR="outs/experiments/text/node/dataset2/hybrid/bert.out" LOG="outs/experiments/text/node/dataset2/hybrid/bert.log"

condor_submit JobBatchName=node_dedicated_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/hybrid/dedicated_bert.yaml --cls DedicatedTextSingleItemRun" OUT="outs/experiments/text/node/dataset2/hybrid/dedicated_bert.out" ERR="outs/experiments/text/node/dataset2/hybrid/dedicated_bert.out" LOG="outs/experiments/text/node/dataset2/hybrid/dedicated_bert.log"

condor_submit JobBatchName=node_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/hybrid/shared_bge.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/dataset2/hybrid/bge.out" ERR="outs/experiments/text/node/dataset2/hybrid/bge.out" LOG="outs/experiments/text/node/dataset2/hybrid/bge.log"

condor_submit JobBatchName=node_dedicated_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/hybrid/dedicated_bge.yaml --cls DedicatedTextSingleItemRun" OUT="outs/experiments/text/node/dataset2/hybrid/dedicated_bge.out" ERR="outs/experiments/text/node/dataset2/hybrid/dedicated_bge.out" LOG="outs/experiments/text/node/dataset2/hybrid/dedicated_bge.log"

condor_submit JobBatchName=node_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/hybrid/shared_modernbert.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/dataset2/hybrid/modernbert.out" ERR="outs/experiments/text/node/dataset2/hybrid/modernbert.out" LOG="outs/experiments/text/node/dataset2/hybrid/modernbert.log"

condor_submit JobBatchName=node_dedicated_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/hybrid/dedicated_modernbert.yaml --cls DedicatedTextSingleItemRun" OUT="outs/experiments/text/node/dataset2/hybrid/dedicated_modernbert.out" ERR="outs/experiments/text/node/dataset2/hybrid/dedicated_modernbert.out" LOG="outs/experiments/text/node/dataset2/hybrid/dedicated_modernbert.log"

condor_submit JobBatchName=node_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/hybrid/shared_roberta.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/dataset2/hybrid/roberta.out" ERR="outs/experiments/text/node/dataset2/hybrid/roberta.out" LOG="outs/experiments/text/node/dataset2/hybrid/roberta.log"

condor_submit JobBatchName=node_dedicated_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/hybrid/dedicated_roberta.yaml --cls DedicatedTextSingleItemRun" OUT="outs/experiments/text/node/dataset2/hybrid/dedicated_roberta.out" ERR="outs/experiments/text/node/dataset2/hybrid/dedicated_roberta.out" LOG="outs/experiments/text/node/dataset2/hybrid/dedicated_roberta.log"