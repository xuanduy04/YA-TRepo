#!/bin/bash

N_NODES=2         # Total nodes, must be >= 2
VLLM_NODES=1      # vLLM nodes, must be < N_NODES, currently fixed at 1
N_GPUS_PER_NODE=4 # Allowed: 4 or 8

MODEL_PATH="Qwen/Qwen3-0.6B-Base"
TOKENIZER_PATH=""

# --- Data, Output and Logging paths --- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"
DATA_PATH="$ROOT_DIR/data_ready2train/*.jsonl" # <--- DATA_PATH HERE

MODEL_NAME="$(basename "$MODEL_PATH")"
RUN_NAME="${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$ROOT_DIR/outputs/consept-$RUN_NAME" # <--- OUTPUT_DIR HERE

VLLM_SERVE_LOG_FILE="$OUTPUT_DIR/vllm_serve_logs-${RUN_NAME}.txt"
TRAINING_LOG_FILE="$OUTPUT_DIR/training_logs-${RUN_NAME}.txt"

# --- exporting hf home and cache, if you need to do so --- #
# export HF_HOME=""
# export HF_DATASETS_CACHE=""
# export HUGGINGFACE_HUB_CACHE=""

# ------------------------------------------------------------------ #

# --- Validate vars --- #
require_var() {
	[ -z "${!1}" ] && {
		echo "Error: $1 must be set." >&2
		exit 1
	}
}
require_var N_NODES
require_var VLLM_NODES
require_var N_GPUS_PER_NODE
require_var MODEL_PATH
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"

# We partition tasks by node-level so all devices are used for a single task.
if ((N_GPUS_PER_NODE == 4)); then
	export CUDA_VISIBLE_DEVICES=0,1,2,3
elif ((N_GPUS_PER_NODE == 8)); then
	export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
	echo "N_GPUS_PER_NODE should be 4 or 8"
	exit 1
fi

# --- Main logic --- #
NODE_ID=$((RANK / N_GPUS_PER_NODE))

if ((N_NODES - VLLM_NODES <= NODE_ID)); then
	# --- Task 1: Serve the model --- #
	echo -e "Log file of 'trl vllm-serve' is at:\n\t'$TRL_VLLM_SERVE_LOG_FILE'"
	bash $SCRIPT_DIR/multinode_grpo_vllm.sh \
        --vllm_nodes VLLM_NODES \
        --n_gpus_per_node N_GPUS_PER_NODE \
        --model_path MODEL_PATH \
        --tokenizer_path TOKENIZER_PATH \
		--served_model_name MODEL_NAME \
		--log_file VLLM_SERVE_LOG_FILE
else
	# --- Task 2: Run the training --- #
	echo -n "Waiting for TRL's vLLM server... "
	until curl -sf "http://0.0.0.0:8000/health" >/dev/null 2>&1; do
		sleep 5
	done
	echo "Done."
	echo -e "Log file of training is at:\n\t'$TRAINING_LOG_FILE'"
	bash $SCRIPT_DIR/multinode_grpo_train.sh \
		--training_nodes $(( N_NODES - VLLM_NODES )) \
		--n_gpus_per_node N_GPUS_PER_NODE \
        --model_path MODEL_PATH \
        --tokenizer_path TOKENIZER_PATH \
		--log_file TRAINING_LOG_FILE
fi

