#!/bin/bash

echo "TEXT NODE DATASET1 HYBRID"
condor_submit JobBatchName=node_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/hybrid/shared_bert.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/hybrid/bert.out" ERR="outs/experiments/text/node/hybrid/bert.out" LOG="outs/experiments/text/node/hybrid/bert.log"

condor_submit JobBatchName=node_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/hybrid/shared_bge.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/hybrid/bge.out" ERR="outs/experiments/text/node/hybrid/bge.out" LOG="outs/experiments/text/node/hybrid/bge.log"

condor_submit JobBatchName=node_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/hybrid/shared_modernbert.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/hybrid/modernbert.out" ERR="outs/experiments/text/node/hybrid/modernbert.out" LOG="outs/experiments/text/node/hybrid/modernbert.log"

condor_submit JobBatchName=node_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/hybrid/shared_roberta.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/hybrid/roberta.out" ERR="outs/experiments/text/node/hybrid/roberta.out" LOG="outs/experiments/text/node/hybrid/roberta.log"