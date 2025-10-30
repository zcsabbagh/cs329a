#!/usr/bin/env python3
"""
Finetune using Together AI API (No GPU Required)

Together AI supports finetuning open-source models like:
- Meta-Llama-3.1-8B-Instruct
- Meta-Llama-3.1-70B-Instruct
- Mistral-7B-Instruct-v0.3

Cost: Much cheaper than OpenAI, typically $5-15 for similar dataset

Usage:
    python finetune_together.py \
        --training-file data/training_data_clean.jsonl \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --suffix ptom-agent

Requirements:
    pip install together
    export TOGETHER_API_KEY=your-key-here
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List
from together import Together


def estimate_cost_together(num_examples: int, model: str) -> float:
    """Estimate Together AI finetuning cost."""

    # Together AI pricing is much cheaper than OpenAI
    # Approximate costs per 1M tokens:
    costs_per_model = {
        "meta-llama/Meta-Llama-3.1-8B-Instruct": 2.0,
        "meta-llama/Meta-Llama-3.1-70B-Instruct": 8.0,
        "mistralai/Mistral-7B-Instruct-v0.3": 2.0,
    }

    cost_per_1m = costs_per_model.get(model, 3.0)  # Default estimate

    # Estimate tokens (same as before)
    avg_tokens_per_example = 500
    total_tokens = num_examples * avg_tokens_per_example

    estimated_cost = (total_tokens / 1_000_000) * cost_per_1m

    return estimated_cost, total_tokens


def monitor_finetuning(client: Together, job_id: str, check_interval: int = 60):
    """Monitor finetuning job until completion."""

    print(f"\n⏳ Monitoring job {job_id}...")
    print(f"  (Checking every {check_interval} seconds)")
    print(f"  This typically takes 30-60 minutes")
    print()

    while True:
        status_data = client.fine_tuning.retrieve(job_id)
        status = status_data.status

        print(f"  Status: {status}")

        if status == "succeeded":
            print(f"\n✅ Training complete!")
            model_name = status_data.output_name
            print(f"📦 Model: {model_name}")
            return model_name

        elif status == "failed":
            print(f"\n❌ Training failed!")
            error = getattr(status_data, 'error', 'Unknown error')
            print(f"Error: {error}")
            return None

        elif status == "cancelled":
            print(f"\n⚠️  Training cancelled")
            return None

        else:
            # Still running (queued, running, etc.)
            time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Finetune using Together AI (no GPU required)"
    )

    parser.add_argument(
        "--training-file",
        type=str,
        required=True,
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        default=None,
        help="Path to validation data JSONL (optional)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model (meta-llama/Meta-Llama-3.1-8B-Instruct, meta-llama/Meta-Llama-3.1-70B-Instruct, mistralai/Mistral-7B-Instruct-v0.3)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="ptom-agent",
        help="Model name suffix"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Don't wait for training to complete"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Together AI API key (default: use TOGETHER_API_KEY env var)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and proceed automatically"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("❌ Error: No API key provided")
        print("Set TOGETHER_API_KEY environment variable or use --api-key")
        print("\nGet your API key at: https://api.together.xyz/settings/api-keys")
        return

    # Initialize client
    client = Together(api_key=api_key)

    print("=" * 70)
    print("Together AI Finetuning (No GPU Required)")
    print("=" * 70)

    # Check if file exists
    training_file = Path(args.training_file)
    if not training_file.exists():
        print(f"❌ Training file not found: {training_file}")
        return

    # Count examples and estimate cost
    with open(training_file, 'r') as f:
        num_examples = sum(1 for _ in f)

    estimated_cost, total_tokens = estimate_cost_together(num_examples, args.model)

    print(f"\n📊 Training Details:")
    print(f"  Examples: {num_examples}")
    print(f"  Estimated tokens: {total_tokens:,}")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")

    # Confirm
    if not args.yes:
        response = input(f"\n⚠️  Proceed with training? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled")
            return
    else:
        print(f"\n✅ Auto-proceeding with training (--yes flag set)")

    # Upload training file
    print(f"\n📤 Uploading training file...")
    training_file_resp = client.files.upload(file=str(training_file), purpose="fine-tune")
    training_file_id = training_file_resp.id
    print(f"✅ Uploaded! File ID: {training_file_id}")

    # Upload validation file if provided
    validation_file_id = None
    if args.validation_file:
        validation_file = Path(args.validation_file)
        if validation_file.exists():
            print(f"\n📤 Uploading validation file...")
            validation_file_resp = client.files.upload(file=str(validation_file), purpose="fine-tune")
            validation_file_id = validation_file_resp.id
            print(f"✅ Uploaded! File ID: {validation_file_id}")

    # Create finetuning job
    print(f"\n🚀 Starting finetuning job...")
    print(f"  Model: {args.model}")
    print(f"  Training file: {training_file_id}")
    if validation_file_id:
        print(f"  Validation file: {validation_file_id}")

    finetune_params = {
        "training_file": training_file_id,
        "model": args.model,
        "suffix": args.suffix,
        "n_epochs": args.epochs,
    }
    if validation_file_id:
        finetune_params["validation_file"] = validation_file_id

    job_resp = client.fine_tuning.create(**finetune_params)
    job_id = job_resp.id

    print(f"✅ Job created! Job ID: {job_id}")
    print(f"\n📊 Monitor at: https://api.together.xyz/playground/fine-tuning/{job_id}")

    # Save job info
    job_info = {
        "job_id": job_id,
        "model": args.model,
        "suffix": args.suffix,
        "training_file": str(training_file),
        "training_file_id": training_file_id,
        "validation_file_id": validation_file_id,
        "num_examples": num_examples,
        "estimated_cost": estimated_cost,
        "provider": "together_ai"
    }

    job_info_file = Path("data/together_finetuning_job_info.json")
    job_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(job_info_file, 'w') as f:
        json.dump(job_info, f, indent=2)

    print(f"\n💾 Job info saved to: {job_info_file}")

    # Monitor if requested
    if not args.no_monitor:
        finetuned_model_id = monitor_finetuning(client, job_id)

        if finetuned_model_id:
            # Save model info
            model_info = {
                "model_id": finetuned_model_id,
                "base_model": args.model,
                "suffix": args.suffix,
                "job_id": job_id,
                "provider": "together_ai"
            }

            model_info_file = Path("data/together_finetuned_model_info.json")
            with open(model_info_file, 'w') as f:
                json.dump(model_info, f, indent=2)

            print(f"\n💾 Model info saved to: {model_info_file}")
            print("\n" + "=" * 70)
            print("✅ Finetuning Complete!")
            print("=" * 70)
            print(f"\nYour model ID: {finetuned_model_id}")
    else:
        print("\n" + "=" * 70)
        print(f"⏳ Training started (running in background)")
        print("=" * 70)
        print(f"\nJob ID: {job_id}")
        print(f"\nCheck status at: https://api.together.xyz/playground/fine-tuning/{job_id}")


if __name__ == "__main__":
    main()
