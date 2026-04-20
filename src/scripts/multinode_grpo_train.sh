#!/bin/bash

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
OUTPUT_DIR="$ROOT_DIR/outputs/GRPO-$RUN_NAME" # <--- OUTPUT_DIR HERE

TRAINING_LOG_FILE="$OUTPUT_DIR/training_logs-${RUN_NAME}.logs"

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
require_var N_NODES
require_var N_GPUS_PER_NODE
require_var MODEL_PATH
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6767}"
VLLM_SERVER_URL="${VLLM_SERVER_URL:-${MASTER_ADDR}:8000}"
((N_NODES > 1)) || {
	echo "N_NODES should be greater than 1, (currently $N_NODES)" && exit 1
}
bash $SCRIPT_DIR/validate_gpu_count.sh $N_GPUS_PER_NODE || exit 1

# --- Main code --- #
bash $SCRIPT_DIR/install_dependancies.sh $ROOT_DIR || exit 1
mkdir -p $OUTPUT_DIR


cd "$SRC_DIR" || exit 1
run_training() {
    echo "Setting up training endpoint at MASTER_ADDR=${MASTER_ADDR} | MASTER_PORT=${MASTER_PORT}..."
    SAFETENSORS_FAST_GPU=1 \
    HF_ENABLE_PARALLEL_LOADING=true HF_PARALLEL_LOADING_WORKERS=$(( 100 / N_GPUS_PER_NODE )) \
    TRL_EXPERIMENTAL_SILENCE=1 PYTHONPATH=$SRC_DIR accelerate launch \
        --config_file accelerate_configs/multinode.yaml \
        --parallelism_config_tp_size $(( N_NODES * N_GPUS_PER_NODE )) \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --num_machines $N_NODES \
        --num_processes $(( N_NODES * N_GPUS_PER_NODE )) \
        main/multinode_grpo.py \
        --model_name_or_path $MODEL_PATH \
        --tokenizer_name_or_path $TOKENIZER_PATH \
        --output_dir $OUTPUT_DIR \
        --dataset_name $DATA_PATH \
        --use_vllm true \
        --vllm_mode 'server' \
        --vllm_server_base_url $VLLM_SERVER_URL \
        --learning_rate 1e-6 \
        --dtype bfloat16 \
        --max_completion_length 128 \
        --max_steps 22 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --num_generations 2 \
        --gradient_checkpointing false \
        --beta 0.0 \
        --warmup_steps 300 \
        --max_grad_norm 1.0 \
        --log_completions true \
        --num_completions_to_print 1 \
        --logging_steps 25 \
        --save_strategy steps \
        --save_steps 2 \
        --report_to tensorboard \
        --trust_remote_code true \
        --seed 2212 \
        2>&1
}

if (( RANK < N_NODES - 1 )); then
    echo -e "Log file of training will be at:\n\t'$TRAINING_LOG_FILE'\n\n"
    run_training | tee $TRAINING_LOG_FILE
else
    # Last node will host the vLLM instance
    MODEL_PATH=$MODEL_PATH bash "$SCRIPT_DIR/multinode_grpo_trl_vllm_serve.sh" || exit 1
fi
