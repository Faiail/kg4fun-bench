#!/bin/bash

mkdir -p outs/experiments/text/node/dataset1/inductive
echo "TEXT CROSS NODE DATASET1 INDUCTIVE"

condor_submit JobBatchName=cross_node_bert_dataset1_inductive ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/cross_bert.yaml --cls SharedCrossTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/cross_bert.out" ERR="outs/experiments/text/node/dataset1/inductive/cross_bert.out" LOG="outs/experiments/text/node/dataset1/inductive/cross_bert.log"

condor_submit JobBatchName=cross_node_bge_dataset1_inductive ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/cross_bge.yaml --cls SharedCrossTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/cross_bge.out" ERR="outs/experiments/text/node/dataset1/inductive/cross_bge.out" LOG="outs/experiments/text/node/dataset1/inductive/cross_bge.log"

condor_submit JobBatchName=cross_node_modernbert_dataset1_inductive ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/cross_modernbert.yaml --cls SharedCrossTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/cross_modernbert.out" ERR="outs/experiments/text/node/dataset1/inductive/cross_modernbert.out" LOG="outs/experiments/text/node/dataset1/inductive/cross_modernbert.log"

condor_submit JobBatchName=cross_node_roberta_dataset1_inductive ./condor/launch ARGS="experiment --parameters=configs/experiment/text/node/dataset1/inductive/cross_roberta.yaml --cls SharedCrossTextSingleItemRun" OUT="outs/experiments/text/node/dataset1/inductive/cross_roberta.out" ERR="outs/experiments/text/node/dataset1/inductive/cross_roberta.out" LOG="outs/experiments/text/node/dataset1/inductive/cross_roberta.log"
