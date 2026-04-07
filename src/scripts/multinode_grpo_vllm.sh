#!/bin/bash

while (($# > 0)); do
	case "$1" in
	--vllm_nodes)
		VLLM_NODES="$2"
		shift 2
		;;
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
	--log_file)
		VLLM_SERVE_LOG_FILE="$2"
		shift 2
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done

# as VLLM_NODES is just 1 for now, we don't add any more logic to this.
# eventually, this code will call ray and what not.

vllm serve \
	--model $MODEL_PATH \
	--tokenizer $TOKENIZER_PATH \
	--serve-model-name $MODEL_NAME \
	--port 8000 \
	--gpu-memory-utilization 0.90 \
	--tensor-parallel-size $N_GPUS_PER_NODE \
	--max-model-len 16384 \
	--no-enable-prefix-caching \
	--async-scheduling \
	--enable-expert-parallel \
	--enable-tokenizer-info-endpoint \
	--trust-remote-code \
	--compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}' \
	2>&1 | tee $VLLM_SERVE_LOG_FILE
