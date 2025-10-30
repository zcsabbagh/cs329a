#!/usr/bin/env python3
"""
Finetune LLaMA models using QLoRA for Planning Theory of Mind.

This script uses Parameter-Efficient Fine-Tuning (PEFT) with QLoRA to
efficiently finetune LLaMA models on a single GPU.

Requirements:
    - GPU with 16GB+ VRAM (A100, RTX 4090, etc.)
    - HuggingFace token with LLaMA access
    - Training data from prepare_training_data.py

Usage:
    python finetune_llama.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --train-data data/training_data.jsonl \
        --val-data data/validation_data.jsonl \
        --output-dir models/ptom-llama-8b \
        --epochs 3 \
        --batch-size 4
"""

import os
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
import wandb


def load_jsonl_data(file_path: Path) -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_chat_template(example: Dict, tokenizer) -> Dict:
    """Format example using the model's chat template."""

    # Extract messages
    messages = example.get('messages', [])

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": formatted}


def setup_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    use_flash_attention: bool = True
):
    """Load model and tokenizer with QLoRA configuration."""

    print(f"📥 Loading model: {model_name}")

    # Quantization config for QLoRA (4-bit)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",  # Required for training
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
    )

    # Prepare for k-bit training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    print(f"✅ Model loaded: {model.get_memory_footprint() / 1e9:.2f} GB")

    return model, tokenizer


def setup_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None
) -> LoraConfig:
    """Configure LoRA adapter."""

    if target_modules is None:
        # Default: target attention and MLP layers
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Finetune LLaMA for Planning Theory of Mind"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (QLoRA)"
    )
    parser.add_argument(
        "--use-flash-attention",
        action="store_true",
        default=True,
        help="Use Flash Attention 2 (faster, requires compatible GPU)"
    )

    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data JSONL"
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/ptom-llama",
        help="Output directory for model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )

    # Logging
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=True,
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="planning-tom-finetuning",
        help="W&B project name"
    )

    args = parser.parse_args()

    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args)
        )

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 70)
    print("Loading Data")
    print("=" * 70)

    train_data = load_jsonl_data(Path(args.train_data))
    print(f"✅ Loaded {len(train_data)} training examples")

    val_data = None
    if args.val_data:
        val_data = load_jsonl_data(Path(args.val_data))
        print(f"✅ Loaded {len(val_data)} validation examples")

    # Setup model and tokenizer
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    model, tokenizer = setup_model_and_tokenizer(
        args.model,
        use_4bit=args.use_4bit,
        use_flash_attention=args.use_flash_attention
    )

    # Setup LoRA
    print("\n" + "=" * 70)
    print("Configuring LoRA")
    print("=" * 70)

    lora_config = setup_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    print("\n" + "=" * 70)
    print("Preparing Datasets")
    print("=" * 70)

    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=train_dataset.column_names
    )

    eval_dataset = None
    if val_data:
        eval_dataset = Dataset.from_list(val_data)
        eval_dataset = eval_dataset.map(
            lambda x: format_chat_template(x, tokenizer),
            remove_columns=eval_dataset.column_names
        )

    print(f"✅ Training dataset ready: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"✅ Validation dataset ready: {len(eval_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        report_to="wandb" if args.use_wandb else "none",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
    )

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 70)

    trainer.train()

    # Save model
    print("\n" + "=" * 70)
    print("Saving Model")
    print("=" * 70)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training args
    with open(output_dir / "training_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"✅ Model saved to: {output_dir}")
    print(f"✅ Adapter size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M parameters")

    # Cleanup
    if args.use_wandb:
        wandb.finish()

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\nModel location: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Test the model locally")
    print(f"  2. Run evaluation: cd ../evaluation && python evaluate_agent.py")


if __name__ == "__main__":
    main()
