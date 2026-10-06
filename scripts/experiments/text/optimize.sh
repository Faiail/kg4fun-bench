#!/bin/bash

mkdir -p outs/experiments/text/test
echo "OPTIMIZE"

condor_submit JobBatchName=optimize ./condor/launch ARGS="optimize --parameters=configs/experiment/text/test/optimize.yaml --cls OptunaOptimizer" OUT="outs/experiments/text/test/optuna_optimize.out" ERR="outs/experiments/text/test/optuna_optimize.out" LOG="outs/experiments/text/test/optuna_optimize.log"
