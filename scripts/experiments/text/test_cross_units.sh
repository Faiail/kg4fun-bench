#!/bin/bash

mkdir -p outs/experiments/text/test/cross_units
echo "LAUNCHING EXPLORATORY CROSS-ENCODING UNIT EXPERIMENTS"

# 1. T-GLU Experiments (Luo et al., MIMIC)
condor_submit JobBatchName=cross_tglu_cosface ./condor/launch \
  ARGS="experiment --parameters=configs/experiment/text/test/cross_units/tglu_cosface.yaml --cls SharedCrossTextSingleItemRun" \
  OUT="outs/experiments/text/test/cross_units/tglu_cosface.out" \
  ERR="outs/experiments/text/test/cross_units/tglu_cosface.out" \
  LOG="outs/experiments/text/test/cross_units/tglu_cosface.log"

condor_submit JobBatchName=cross_tglu_infonce ./condor/launch \
  ARGS="experiment --parameters=configs/experiment/text/test/cross_units/tglu_infonce.yaml --cls SharedCrossTextSingleItemRun" \
  OUT="outs/experiments/text/test/cross_units/tglu_infonce.out" \
  ERR="outs/experiments/text/test/cross_units/tglu_infonce.out" \
  LOG="outs/experiments/text/test/cross_units/tglu_infonce.log"

# 2. ColBERT MaxSim Experiments (Khattab & Zaharia)
condor_submit JobBatchName=cross_colbert_cosface ./condor/launch \
  ARGS="experiment --parameters=configs/experiment/text/test/cross_units/colbert_cosface.yaml --cls SharedCrossTextSingleItemRun" \
  OUT="outs/experiments/text/test/cross_units/colbert_cosface.out" \
  ERR="outs/experiments/text/test/cross_units/colbert_cosface.out" \
  LOG="outs/experiments/text/test/cross_units/colbert_cosface.log"

condor_submit JobBatchName=cross_colbert_infonce ./condor/launch \
  ARGS="experiment --parameters=configs/experiment/text/test/cross_units/colbert_infonce.yaml --cls SharedCrossTextSingleItemRun" \
  OUT="outs/experiments/text/test/cross_units/colbert_infonce.out" \
  ERR="outs/experiments/text/test/cross_units/colbert_infonce.out" \
  LOG="outs/experiments/text/test/cross_units/colbert_infonce.log"

# 3. Enhanced Bidirectional + MaxSim Experiments
condor_submit JobBatchName=cross_enhanced_cosface ./condor/launch \
  ARGS="experiment --parameters=configs/experiment/text/test/cross_units/enhanced_cosface.yaml --cls SharedCrossTextSingleItemRun" \
  OUT="outs/experiments/text/test/cross_units/enhanced_cosface.out" \
  ERR="outs/experiments/text/test/cross_units/enhanced_cosface.out" \
  LOG="outs/experiments/text/test/cross_units/enhanced_cosface.log"

condor_submit JobBatchName=cross_enhanced_infonce ./condor/launch \
  ARGS="experiment --parameters=configs/experiment/text/test/cross_units/enhanced_infonce.yaml --cls SharedCrossTextSingleItemRun" \
  OUT="outs/experiments/text/test/cross_units/enhanced_infonce.out" \
  ERR="outs/experiments/text/test/cross_units/enhanced_infonce.out" \
  LOG="outs/experiments/text/test/cross_units/enhanced_infonce.log"

