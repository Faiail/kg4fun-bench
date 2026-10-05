#!/bin/bash

mkdir -p outs/experiments/text/test
echo "LAUNCHING TEXT TEST HYPERPARAMETER EXPERIMENTS"

# base
condor_submit JobBatchName=test_base ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/base.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/base.out" ERR="outs/experiments/text/test/base.out" LOG="outs/experiments/text/test/base.log"

# Temperature Sweeps
condor_submit JobBatchName=test_temp_0.02 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/temp_0.02.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/temp_0.02.out" ERR="outs/experiments/text/test/temp_0.02.out" LOG="outs/experiments/text/test/temp_0.02.log"

condor_submit JobBatchName=test_temp_0.05 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/temp_0.05.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/temp_0.05.out" ERR="outs/experiments/text/test/temp_0.05.out" LOG="outs/experiments/text/test/temp_0.05.log"

condor_submit JobBatchName=test_temp_0.07 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/temp_0.07.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/temp_0.07.out" ERR="outs/experiments/text/test/temp_0.07.out" LOG="outs/experiments/text/test/temp_0.07.log"

condor_submit JobBatchName=test_temp_0.10 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/temp_0.10.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/temp_0.10.out" ERR="outs/experiments/text/test/temp_0.10.out" LOG="outs/experiments/text/test/temp_0.10.log"

# Projector Hidden Size Sweeps
condor_submit JobBatchName=test_proj_128 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/proj_128.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/proj_128.out" ERR="outs/experiments/text/test/proj_128.out" LOG="outs/experiments/text/test/proj_128.log"

condor_submit JobBatchName=test_proj_256 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/proj_256.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/proj_256.out" ERR="outs/experiments/text/test/proj_256.out" LOG="outs/experiments/text/test/proj_256.log"

condor_submit JobBatchName=test_proj_512 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/proj_512.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/proj_512.out" ERR="outs/experiments/text/test/proj_512.out" LOG="outs/experiments/text/test/proj_512.log"

# Batch Size Sweeps
condor_submit JobBatchName=test_batch_256 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/batch_256.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/batch_256.out" ERR="outs/experiments/text/test/batch_256.out" LOG="outs/experiments/text/test/batch_256.log"

condor_submit JobBatchName=test_batch_1024 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/batch_1024.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/batch_1024.out" ERR="outs/experiments/text/test/batch_1024.out" LOG="outs/experiments/text/test/batch_1024.log"

# Unfrozen Layers Sweeps
condor_submit JobBatchName=test_unfrozen_top2 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/unfrozen_top2.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/unfrozen_top2.out" ERR="outs/experiments/text/test/unfrozen_top2.out" LOG="outs/experiments/text/test/unfrozen_top2.log"

condor_submit JobBatchName=test_unfrozen_top4 ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/unfrozen_top4.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/unfrozen_top4.out" ERR="outs/experiments/text/test/unfrozen_top4.out" LOG="outs/experiments/text/test/unfrozen_top4.log"

condor_submit JobBatchName=test_unfrozen_all ./condor/launch ARGS="experiment --parameters=configs/experiment/text/test/unfrozen_all.yaml --cls SharedTextSingleItemRun" OUT="outs/experiments/text/test/unfrozen_all.out" ERR="outs/experiments/text/test/unfrozen_all.out" LOG="outs/experiments/text/test/unfrozen_all.log"
