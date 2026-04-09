#!/bin/bash

while (($# > 0)); do
	case "$1" in
	--n_gpus_per_node)
		N_GPUS_PER_NODE="$2"
		shift 2
		;;
	--model_path)
		MODEL_PATH="$2"
		shift 2
		;;
	--tokenizer_path)
		TOKENIZER_PATH="$2"
		shift 2
		;;
	--served_model_name)
		MODEL_NAME="$2"
		shift 2
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done

TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
MODEL_NAME="${MODEL_NAME:-$(basename $MODEL_PATH)}"

VLLM_SERVER_DEV_MODE=1 vllm serve $MODEL_PATH \
	--tokenizer $TOKENIZER_PATH \
	--served-model-name $MODEL_NAME \
	--port 8000 \
	--gpu-memory-utilization 0.90 \
	--tensor-parallel-size $N_GPUS_PER_NODE \
	--max-model-len 16384 \
	--no-enable-prefix-caching \
	--async-scheduling \
	--enable-tokenizer-info-endpoint \
	--trust-remote-code \
	--compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}' \
	--logprobs-mode processed_logprobs \
	--weight-transfer-config '{"backend":"nccl"}'

# --enable-expert-parallel \
