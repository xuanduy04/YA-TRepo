import logging

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
)
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.rewards import accuracy_reward


DEBUG = False
if DEBUG:
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, AsyncGRPOConfig, ModelConfig))
    script_args, training_args, model_args, reward_args = parser.parse_args_and_config()
    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    processor = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left")

    ################
    # Dataset
    ################
    train_dataset = load_dataset(
        "json",
        data_files=script_args.dataset_name,
        split="train",
        streaming=script_args.dataset_streaming,
        num_proc=None if script_args.dataset_streaming else 67,
    )
    # train_dataset = train_dataset.select_columns(["text"])

    ################
    # Training
    ################
    trainer = AsyncGRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=processor,
        args=training_args,
        reward_funcs=accuracy_reward,
        train_dataset=train_dataset,
    )
    # validate_accelerator_config(trainer.accelerator)
    trainer.accelerator.print(f"Begin training {trainer._name} for model `{model_args.model_name_or_path}`")
    resume_from_checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint at '{resume_from_checkpoint}'")
    trainer.train(resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
