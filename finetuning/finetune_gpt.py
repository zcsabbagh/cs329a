#!/usr/bin/env python3
"""
Finetune GPT-4o-mini using OpenAI API (No GPU Required)

This script uploads your training data to OpenAI and starts finetuning.
Training happens on OpenAI's servers, so no local GPU needed!

Cost: ~$8 per 1M tokens (typically $10-25 for 2000 examples)

Usage:
    python finetune_gpt.py \
        --training-file data/openai_training_data.jsonl \
        --model gpt-4o-mini-2024-07-18 \
        --suffix ptom-agent

Requirements:
    pip install openai
    export OPENAI_API_KEY=sk-xxx
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List
import openai


def convert_to_openai_format(input_file: Path, output_file: Path):
    """
    Convert ChatML format to OpenAI finetuning format.

    OpenAI format requires:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """

    print(f"📂 Converting {input_file} to OpenAI format...")

    examples = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Check if already in correct format
            if 'messages' in data:
                # Extract just the messages
                messages = data['messages']
                examples.append({"messages": messages})
            else:
                print(f"⚠️  Unexpected format: {data.keys()}")

    # Write to output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Converted {len(examples)} examples")
    print(f"💾 Saved to: {output_file}")

    return examples


def estimate_cost(num_examples: int, avg_tokens_per_example: int = 500):
    """Estimate training cost."""

    total_tokens = num_examples * avg_tokens_per_example

    # OpenAI pricing (as of 2024)
    cost_per_1m_tokens = 8.00  # $8 per 1M tokens for training

    estimated_cost = (total_tokens / 1_000_000) * cost_per_1m_tokens

    return estimated_cost, total_tokens


def upload_training_file(client: openai.OpenAI, file_path: Path) -> str:
    """Upload training file to OpenAI."""

    print(f"\n📤 Uploading training file...")

    with open(file_path, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose='fine-tune'
        )

    file_id = response.id
    print(f"✅ Uploaded! File ID: {file_id}")

    return file_id


def create_finetuning_job(
    client: openai.OpenAI,
    training_file_id: str,
    model: str,
    suffix: str,
    validation_file_id: str = None,
    hyperparameters: Dict = None
) -> str:
    """Create a finetuning job."""

    print(f"\n🚀 Starting finetuning job...")
    print(f"  Model: {model}")
    print(f"  Training file: {training_file_id}")
    if validation_file_id:
        print(f"  Validation file: {validation_file_id}")

    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            "n_epochs": 3,
        }

    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        suffix=suffix,
        hyperparameters=hyperparameters
    )

    job_id = response.id
    print(f"✅ Job created! Job ID: {job_id}")
    print(f"\n📊 Monitor at: https://platform.openai.com/finetune/{job_id}")

    return job_id


def monitor_finetuning_job(client: openai.OpenAI, job_id: str, check_interval: int = 60):
    """Monitor finetuning job until completion."""

    print(f"\n⏳ Monitoring job {job_id}...")
    print(f"  (Checking every {check_interval} seconds)")
    print(f"  This typically takes 1-2 hours")
    print()

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        print(f"  Status: {status}")

        if status == "succeeded":
            print(f"\n✅ Training complete!")
            print(f"📦 Model: {job.fine_tuned_model}")
            return job.fine_tuned_model

        elif status == "failed":
            print(f"\n❌ Training failed!")
            print(f"Error: {job.error}")
            return None

        elif status == "cancelled":
            print(f"\n⚠️  Training cancelled")
            return None

        else:
            # Still running
            time.sleep(check_interval)


def test_finetuned_model(client: openai.OpenAI, model_id: str):
    """Quick test of the finetuned model."""

    print(f"\n🧪 Testing finetuned model...")

    test_messages = [
        {
            "role": "system",
            "content": "You are an expert advocate using Planning Theory of Mind."
        },
        {
            "role": "user",
            "content": "The target is considering two vacation options. What should you do first?"
        }
    ]

    response = client.chat.completions.create(
        model=model_id,
        messages=test_messages,
        max_tokens=200
    )

    print(f"✅ Test response:")
    print(f"{response.choices[0].message.content}")


def main():
    parser = argparse.ArgumentParser(
        description="Finetune GPT using OpenAI API (no GPU required)"
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
        default="gpt-4o-mini-2024-07-18",
        help="Base model (gpt-4o-mini-2024-07-18 or gpt-4o-2024-08-06)"
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
        help="OpenAI API key (default: use OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and proceed automatically"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: No API key provided")
        print("Set OPENAI_API_KEY environment variable or use --api-key")
        return

    # Initialize client
    client = openai.OpenAI(api_key=api_key)

    print("=" * 70)
    print("OpenAI GPT Finetuning (No GPU Required)")
    print("=" * 70)

    # Check if file needs conversion
    training_file = Path(args.training_file)
    if not training_file.exists():
        print(f"❌ Training file not found: {training_file}")
        return

    # Count examples and estimate cost
    with open(training_file, 'r') as f:
        num_examples = sum(1 for _ in f)

    estimated_cost, total_tokens = estimate_cost(num_examples)

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
    training_file_id = upload_training_file(client, training_file)

    # Upload validation file if provided
    validation_file_id = None
    if args.validation_file:
        validation_file = Path(args.validation_file)
        if validation_file.exists():
            validation_file_id = upload_training_file(client, validation_file)

    # Create finetuning job
    job_id = create_finetuning_job(
        client,
        training_file_id,
        args.model,
        args.suffix,
        validation_file_id,
        hyperparameters={"n_epochs": args.epochs}
    )

    # Save job info
    job_info = {
        "job_id": job_id,
        "model": args.model,
        "suffix": args.suffix,
        "training_file": str(training_file),
        "training_file_id": training_file_id,
        "validation_file_id": validation_file_id,
        "num_examples": num_examples,
        "estimated_cost": estimated_cost
    }

    job_info_file = Path("data/finetuning_job_info.json")
    job_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(job_info_file, 'w') as f:
        json.dump(job_info, f, indent=2)

    print(f"\n💾 Job info saved to: {job_info_file}")

    # Monitor if requested
    if not args.no_monitor:
        finetuned_model_id = monitor_finetuning_job(client, job_id)

        if finetuned_model_id:
            # Test model
            test_finetuned_model(client, finetuned_model_id)

            # Save model info
            model_info = {
                "model_id": finetuned_model_id,
                "base_model": args.model,
                "suffix": args.suffix,
                "job_id": job_id
            }

            model_info_file = Path("data/finetuned_model_info.json")
            with open(model_info_file, 'w') as f:
                json.dump(model_info, f, indent=2)

            print(f"\n💾 Model info saved to: {model_info_file}")
            print("\n" + "=" * 70)
            print("✅ Finetuning Complete!")
            print("=" * 70)
            print(f"\nYour model ID: {finetuned_model_id}")
            print(f"\nNext steps:")
            print(f"  cd ../evaluation")
            print(f"  python evaluate_agent.py --model {finetuned_model_id}")
    else:
        print("\n" + "=" * 70)
        print(f"⏳ Training started (running in background)")
        print("=" * 70)
        print(f"\nJob ID: {job_id}")
        print(f"\nCheck status:")
        print(f"  python finetune_gpt.py --check-job {job_id}")
        print(f"\nOr visit: https://platform.openai.com/finetune/{job_id}")


if __name__ == "__main__":
    main()
