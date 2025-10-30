# LLaMA Finetuning for Planning Theory of Mind

This directory contains scripts for finetuning LLaMA models on the Planning ToM persuasion trajectories.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (required for LLaMA access)
export HUGGINGFACE_TOKEN=hf_xxx

# Optional: Set WandB for experiment tracking
export WANDB_API_KEY=xxx
```

## Quick Start

### Step 1: Prepare Training Data
```bash
python prepare_training_data.py \
  --input ../business_vacation_traj.jsonl \
  --output data/training_data.jsonl \
  --format chatml \
  --train-split 0.9
```

This will create:
- `data/training_data.jsonl` - 90% for training
- `data/validation_data.jsonl` - 10% for validation

### Step 2: Finetune Model
```bash
python finetune_llama.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train-data data/training_data.jsonl \
  --val-data data/validation_data.jsonl \
  --output-dir models/ptom-llama-8b \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-4
```

Training takes ~2-4 hours on a single A100 GPU.

## Model Options

- `meta-llama/Meta-Llama-3.1-8B-Instruct` (Recommended)
- `meta-llama/Meta-Llama-3.1-70B-Instruct` (Better quality, requires multi-GPU)
- `mistralai/Mistral-7B-Instruct-v0.3` (Alternative)

## Output

The finetuned model will be saved as a LoRA adapter in:
- `models/ptom-llama-8b/` - Adapter weights (~50MB)
- `models/ptom-llama-8b/config.json` - Configuration
- `models/ptom-llama-8b/training_args.json` - Training hyperparameters

## Next Steps

After training, go to the `../evaluation/` directory to evaluate your model.
