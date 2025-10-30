#!/usr/bin/env python3
"""
Clean training data to only include 'messages' field for OpenAI API.
"""

import json
import sys
from pathlib import Path

def clean_training_file(input_file: Path, output_file: Path):
    """Remove extra fields, keep only 'messages'."""

    print(f"📂 Cleaning {input_file}...")

    cleaned_count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)

            # Keep only the messages field
            if 'messages' in data:
                cleaned = {"messages": data['messages']}
                outfile.write(json.dumps(cleaned) + '\n')
                cleaned_count += 1
            else:
                print(f"⚠️  Warning: No 'messages' field found in example")

    print(f"✅ Cleaned {cleaned_count} examples")
    print(f"💾 Saved to: {output_file}")

    return cleaned_count

if __name__ == "__main__":
    # Clean training data
    training_in = Path("/Users/irawadee.t/Desktop/classes/ptom-bench/cs329a/finetuning/data/training_data.jsonl")
    training_out = Path("/Users/irawadee.t/Desktop/classes/ptom-bench/cs329a/finetuning/data/training_data_clean.jsonl")

    validation_in = Path("/Users/irawadee.t/Desktop/classes/ptom-bench/cs329a/finetuning/data/validation_data.jsonl")
    validation_out = Path("/Users/irawadee.t/Desktop/classes/ptom-bench/cs329a/finetuning/data/validation_data_clean.jsonl")

    print("=" * 70)
    print("Cleaning Training Data for OpenAI API")
    print("=" * 70)
    print()

    train_count = clean_training_file(training_in, training_out)
    print()
    val_count = clean_training_file(validation_in, validation_out)

    print()
    print("=" * 70)
    print("✅ Cleaning Complete!")
    print("=" * 70)
    print(f"\nTraining examples: {train_count}")
    print(f"Validation examples: {val_count}")
    print(f"\nUse these files for finetuning:")
    print(f"  --training-file {training_out}")
    print(f"  --validation-file {validation_out}")
