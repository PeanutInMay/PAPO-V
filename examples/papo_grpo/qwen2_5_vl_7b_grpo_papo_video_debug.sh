#!/bin/bash

set -x

CUDA_IDS=0,1,2,3
N_GPU=4

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/home/zhzhu/model/Qwen2.5-VL-7B-Instruct   

TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=128    # reduced vs image (384) due to long video sequences
VAL_BATCH_SIZE=256
MAX_PROMPT_LENGTH=8192    # video prompts need longer context

EXP_NAME="qwen2_5_vl_7b__grpo__papo__video__ep${TOTAL_EPOCHES}_rb${ROLLOUT_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)"

CONFIG_FILE="examples/configs/config_grpo_papo_video_7b_debug.yaml"
TRAIN_FILE="/home/zhzhu/dataset/Video-R1-data/PerceptionTest_train_parquet/"   
VAL_FILE="/home/zhzhu/dataset/Video-R1-data/PerceptionTest_val_parquet/"      

FORMAT_PROMPT="examples/format_prompt/math_perception.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score"

KL_PRCP_COEF=0.02

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONFIG_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    algorithm.kl_prcp_coef=${KL_PRCP_COEF}
