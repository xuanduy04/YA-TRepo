#!/bin/bash

VLLM_SERVER_URL=""
N_NODES=2
N_GPUS_PER_NODE=4

MODEL_PATH="Qwen/Qwen3-0.6B"
TOKENIZER_PATH=""

# --- Data, Output and Logging paths --- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"
DATA_PATH="$ROOT_DIR/data_ready2train/*.jsonl" # <--- DATA_PATH HERE

MODEL_NAME="$(basename "$MODEL_PATH")"
RUN_NAME="${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$ROOT_DIR/outputs/consept-$RUN_NAME" # <--- OUTPUT_DIR HERE

TRAINING_LOG_FILE="$OUTPUT_DIR/training_logs-${RUN_NAME}.txt"

# --- exporting hf home and cache, if you need to do so --- #
# export HF_HOME=""
# export HF_DATASETS_CACHE=""
# export HUGGINGFACE_HUB_CACHE=""

# ------------------------------------------------------------------ #

# --- Validate vars --- #
require_var() {
	[ -z "${!1}" ] && {
		echo "Error: variable $1 must be set." >&2
		exit 1
	}
}
require_var VLLM_SERVER_URL
require_var N_NODES
require_var N_GPUS_PER_NODE
require_var MODEL_PATH
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
((N_NODES > 0)) || {
	echo "N_NODES should be a non negative integer, (currently $N_NODES)" && exit 1
}

# --- Main code --- #
bash $SCRIPT_DIR/install_dependancies.sh $ROOT_DIR || exit 1
mkdir -p $OUTPUT_DIR

if command -v curl >/dev/null 2>&1; then
    s=$SECONDS
    until curl -ksf "${VLLM_SERVER_URL}/health" >/dev/null 2>&1; do
        echo "Waiting for vLLM server at ${VLLM_SERVER_URL}... $((SECONDS - s))s"
        sleep 1
    done
    echo "vLLM server is up."
else
    echo "[WARNING]: 'curl' not found. Skipping vLLM server health check..."
fi

echo -e "Log file of training will be at:\n\t'$TRAINING_LOG_FILE'\n\n"

# --main_process_ip $MASTER_ADDR \
# --main_process_port $MASTER_PORT \

cd "$SRC_DIR" || exit 1
PYTHONPATH=$SRC_DIR accelerate launch \
    --config_file accelerate_configs/multinode.yaml \
    --num_machines $N_NODES \
    --num_processes $(( N_NODES * N_GPUS_PER_NODE )) \
    main/multinode_grpo.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATA_PATH \
    --dataset_streaming false \
    --vllm_server_base_url $VLLM_SERVER_URL \
    --learning_rate 1e-6 \
    --dtype bfloat16 \
    --max_completion_length 2048 \
    --max_steps 6767 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 8 \
    --gradient_checkpointing false \
    --warmup_steps 300 \
    --max_grad_norm 1.0 \
    --log_completions true \
    --num_completions_to_print 1 \
    --logging_steps 25 \
    --save_strategy steps \
    --save_steps 50 \
    --report_to tensorboard \
    --seed 2212 \
    2>&1 | tee $TRAINING_LOG_FILE
