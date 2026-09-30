#!/bin/bash


#!/bin/bash

echo "TEXT NODE DATASET2 INDUCTIVE"
condor_submit JobBatchName=node_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/inductive/shared_bert.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/inductive/bert.out" ERR="outs/experiments/text/node/inductive/bert.out" LOG="outs/experiments/text/node/inductive/bert.log"

condor_submit JobBatchName=node_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/inductive/shared_bge.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/inductive/bge.out" ERR="outs/experiments/text/node/inductive/bge.out" LOG="outs/experiments/text/node/inductive/bge.log"

condor_submit JobBatchName=node_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/inductive/shared_modernbert.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/inductive/modernbert.out" ERR="outs/experiments/text/node/inductive/modernbert.out" LOG="outs/experiments/text/node/inductive/modernbert.log"

condor_submit JobBatchName=node_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset2/inductive/shared_roberta.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/node/inductive/roberta.out" ERR="outs/experiments/text/node/inductive/roberta.out" LOG="outs/experiments/text/node/inductive/roberta.log"