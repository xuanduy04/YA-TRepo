#!/bin/bash

while (($# > 0)); do
	case "$1" in
	--training_nodes)
		TRAINING_NODES="$2"
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
	--log_file)
		TRAINING_LOG_FILE="$2"
		shift 2
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done

# ------------------------------------------------------------------ #
run_training() {
    cd "$SRC_DIR" || exit 1
    PYTHONPATH=$SRC_DIR accelerate launch \
		--num_machines $TRAINING_NODES \
        --num_processes $(( TRAINING_NODES * N_GPUS_PER_NODE)) \
        --config_file accelerate_configs/multinode.yaml \
        main/consept_main.py \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_name "$DATA_PATH" \
        --dataset_streaming false \
        --use_vllm \
        --vllm_mode server \
        --vllm_server_port "$MODEL_PORT" \
        --loss_type dr_grpo \
        --learning_rate 1e-6 \
        --dtype bfloat16 \
        --max_completion_length 4000 \
        --max_length 4096 \
        --max_steps 15000 \
        --effective_batch_size 32 \
        --per_device_train_batch_size 8 \
        --num_generations 8 \
        --gradient_checkpointing false \
        --beta 0.0 \
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
}
run_training
