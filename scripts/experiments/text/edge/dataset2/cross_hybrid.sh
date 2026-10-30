#!/bin/bash

mkdir -p outs/experiments/text/edge/dataset2/hybrid
echo "TEXT CROSS EDGE DATASET2 HYBRID"

condor_submit JobBatchName=cross_edge_bert_dataset2_hybrid ./condor/launch ARGS="experiment --parameters=configs/experiment/text/edge/dataset2/hybrid/cross_bert.yaml --cls SharedCrossTextSingleItemRun" OUT="outs/experiments/text/edge/dataset2/hybrid/cross_bert.out" ERR="outs/experiments/text/edge/dataset2/hybrid/cross_bert.out" LOG="outs/experiments/text/edge/dataset2/hybrid/cross_bert.log"

condor_submit JobBatchName=cross_edge_bge_dataset2_hybrid ./condor/launch ARGS="experiment --parameters=configs/experiment/text/edge/dataset2/hybrid/cross_bge.yaml --cls SharedCrossTextSingleItemRun" OUT="outs/experiments/text/edge/dataset2/hybrid/cross_bge.out" ERR="outs/experiments/text/edge/dataset2/hybrid/cross_bge.out" LOG="outs/experiments/text/edge/dataset2/hybrid/cross_bge.log"

condor_submit JobBatchName=cross_edge_modernbert_dataset2_hybrid ./condor/launch ARGS="experiment --parameters=configs/experiment/text/edge/dataset2/hybrid/cross_modernbert.yaml --cls SharedCrossTextSingleItemRun" OUT="outs/experiments/text/edge/dataset2/hybrid/cross_modernbert.out" ERR="outs/experiments/text/edge/dataset2/hybrid/cross_modernbert.out" LOG="outs/experiments/text/edge/dataset2/hybrid/cross_modernbert.log"

condor_submit JobBatchName=cross_edge_roberta_dataset2_hybrid ./condor/launch ARGS="experiment --parameters=configs/experiment/text/edge/dataset2/hybrid/cross_roberta.yaml --cls SharedCrossTextSingleItemRun" OUT="outs/experiments/text/edge/dataset2/hybrid/cross_roberta.out" ERR="outs/experiments/text/edge/dataset2/hybrid/cross_roberta.out" LOG="outs/experiments/text/edge/dataset2/hybrid/cross_roberta.log"
