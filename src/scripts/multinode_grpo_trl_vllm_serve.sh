#!/bin/bash

TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename $MODEL_PATH)}"

# vllm serve $MODEL_PATH \
# 	--tokenizer $TOKENIZER_PATH \
# 	--served-model-name $SERVED_MODEL_NAME \
# 	--gpu-memory-utilization 0.9 \
# 	--trust-remote-code \
# 	--tensor-parallel-size $N_GPUS_PER_NODE \
# 	--data-parallel-size 1 \
# 	--pipeline-parallel-size 1 \
# 	--max-num-batched-tokens 16386 \
# 	--kv-cache-dtype auto \
# 	--port 8000 \
# 	--enable-expert-parallel \
# 	--max-model-len auto \
# 	--logprobs-mode processed_logprobs \
# 	--weight-transfer-config '{"backend":"nccl"}'


trl vllm-serve \
	--model $MODEL_PATH \
	--tokenizer $TOKENIZER_PATH \
	--gpu-memory-utilization 0.92 \
	--trust-remote-code \
	--tensor-parallel-size 4 \
	--data-parallel-size 1 \
	--max-model-len 32768 \
	--kv-cache-dtype auto \
	--host "${MASTER_ADDR:-localhost}" \
	--port 8000 \
	|| exit 1
