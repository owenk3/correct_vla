#!/bin/bash
# Run correctvla eval
# Usage: bash run_eval.sh [ours|llm_baseline] [episode] [max_attempts]

EVAL=${1:-ours}
EPISODE=${2:-127}
MAX_ATTEMPTS=${3:-3}

BASE="CUDA_VISIBLE_DEVICES=1 PYTHONPATH=\$PYTHONPATH:\$(pwd)/LIBERO python experiments/robot/libero/vla_eval_failures.py \
  --pretrained_checkpoint lerobot/pi05_libero_finetuned --model_family pi05 \
  --task_suite_name libero_90 --num_trials_per_task 5"

echo "Running eval=$EVAL episode=$EPISODE max_attempts=$MAX_ATTEMPTS"
eval "$BASE --eval $EVAL --episode $EPISODE --max_attempts $MAX_ATTEMPTS"
