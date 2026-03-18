#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)/LIBERO

CMD="python experiments/robot/libero/vla_eval_failures.py --pretrained_checkpoint lerobot/pi05_libero_finetuned --model_family pi05 --eval llm_baseline --use_model_server True --num_recovery_trials 1 --num_trials_per_task 50 --failure_dir ./rollouts/pi-libero-in/failures"

# libero_spatial
for ep in $(seq 251 300); do $CMD --task_suite_name libero_spatial --episode $ep; done
for ep in $(seq 401 450); do $CMD --task_suite_name libero_spatial --episode $ep; done
for ep in $(seq 451 500); do $CMD --task_suite_name libero_spatial --episode $ep; done

# libero_goal
for ep in $(seq 1 50); do $CMD --task_suite_name libero_goal --episode $ep; done
for ep in $(seq 151 200); do $CMD --task_suite_name libero_goal --episode $ep; done
for ep in $(seq 201 250); do $CMD --task_suite_name libero_goal --episode $ep; done
for ep in $(seq 451 500); do $CMD --task_suite_name libero_goal --episode $ep; done

# libero_10
for ep in $(seq 151 200); do $CMD --task_suite_name libero_10 --episode $ep; done
for ep in $(seq 451 500); do $CMD --task_suite_name libero_10 --episode $ep; done
