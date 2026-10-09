#!/bin/bash

mkdir -p outs/experiments/text/complete/dataset1/hybrid
echo "TEXT COMPLETE DATASET1 HYBRID"
condor_submit JobBatchName=complete_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/shared_bert.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/bert.out" ERR="outs/experiments/text/complete/dataset1/hybrid/bert.out" LOG="outs/experiments/text/complete/dataset1/hybrid/bert.log"

condor_submit JobBatchName=complete_dedicated_bert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/dedicated_bert.yaml --cls DedicatedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/dedicated_bert.out" ERR="outs/experiments/text/complete/dataset1/hybrid/dedicated_bert.out" LOG="outs/experiments/text/complete/dataset1/hybrid/dedicated_bert.log"

condor_submit JobBatchName=complete_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/shared_bge.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/bge.out" ERR="outs/experiments/text/complete/dataset1/hybrid/bge.out" LOG="outs/experiments/text/complete/dataset1/hybrid/bge.log"

condor_submit JobBatchName=complete_dedicated_bge ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/dedicated_bge.yaml --cls DedicatedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/dedicated_bge.out" ERR="outs/experiments/text/complete/dataset1/hybrid/dedicated_bge.out" LOG="outs/experiments/text/complete/dataset1/hybrid/dedicated_bge.log"

condor_submit JobBatchName=complete_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/shared_modernbert.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/modernbert.out" ERR="outs/experiments/text/complete/dataset1/hybrid/modernbert.out" LOG="outs/experiments/text/complete/dataset1/hybrid/modernbert.log"

condor_submit JobBatchName=complete_dedicated_modernbert ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/dedicated_modernbert.yaml --cls DedicatedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/dedicated_modernbert.out" ERR="outs/experiments/text/complete/dataset1/hybrid/dedicated_modernbert.out" LOG="outs/experiments/text/complete/dataset1/hybrid/dedicated_modernbert.log"

condor_submit JobBatchName=complete_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/shared_roberta.yaml --cls SharedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/roberta.out" ERR="outs/experiments/text/complete/dataset1/hybrid/roberta.out" LOG="outs/experiments/text/complete/dataset1/hybrid/roberta.log"

condor_submit JobBatchName=complete_dedicated_roberta ./condor/launch ARGS="experiment --parameters=configs/experiment/text/complete/dataset1/hybrid/dedicated_roberta.yaml --cls DedicatedTextCompleteRun" OUT="outs/experiments/text/complete/dataset1/hybrid/dedicated_roberta.out" ERR="outs/experiments/text/complete/dataset1/hybrid/dedicated_roberta.out" LOG="outs/experiments/text/complete/dataset1/hybrid/dedicated_roberta.log"

