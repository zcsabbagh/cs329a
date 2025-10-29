# Complete Setup Guide: Finetuning & Evaluating Planning ToM Persuasion Agents

## Overview

This guide shows how to:

1. Convert trajectory data to training format
2. Finetune LLaMA or GPT models
3. Create a simulated human agent
4. Evaluate persuasion performance

---

## Part 1: Required Packages & APIs

### Python Packages

```bash
# Core ML/Training
pip install torch transformers datasets accelerate bitsandbytes
pip install peft trl  # For LoRA/QLoRA finetuning
pip install unsloth  # Optional: 2x faster finetuning for LLaMA

# For OpenAI GPT finetuning
pip install openai

# For Anthropic Claude as simulated human
pip install anthropic

# Utilities
pip install wandb  # For experiment tracking
pip install pandas numpy tqdm
pip install scikit-learn  # For evaluation metrics
```

### Required API Keys

**Option A: Finetune LLaMA (Open Source)**

- **HuggingFace Token**: For downloading models
  - Get from: https://huggingface.co/settings/tokens
  - Set: `export HUGGINGFACE_TOKEN=hf_xxx`
- **Weights & Biases (optional)**: For tracking
  - Get from: https://wandb.ai/authorize
  - Set: `export WANDB_API_KEY=xxx`

**Option B: Finetune GPT (OpenAI)**

- **OpenAI API Key**: Required
  - Get from: https://platform.openai.com/api-keys
  - Set: `export OPENAI_API_KEY=sk-xxx`
  - **Cost**: ~$0.008/1K tokens (training) + inference costs

**For Evaluation (Simulated Human)**

- **Anthropic API Key**: To simulate human responses
  - Already have: API KEY

---

## Part 2: Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│                                                              │
│  Trajectories  →  Format Data  →  Finetune  →  Save Model  │
│  (2000 convs)     (turn-based)     (LoRA)      (adapter)    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Evaluation Pipeline                        │
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │  Finetuned   │  ←───→  │  Simulated   │                 │
│  │  Agent       │  turns  │  Human       │                 │
│  │ (Persuader)  │         │  (Claude)    │                 │
│  └──────────────┘         └──────────────┘                 │
│         │                        │                          │
│         └────────┬───────────────┘                          │
│                  ▼                                           │
│          Evaluation Metrics                                 │
│          (success rate, turns)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 3: Data Preparation

### Convert Trajectories to Training Format

Two approaches:

**Approach 1: Supervised Finetuning (SFT)**
Format each turn as prompt → action pairs:

```python
# System message with scenario context + current belief state
# User message: conversation history
# Assistant message: reasoning + counterfactual + action
```

**Approach 2: Reinforcement Learning**
Use success/failure outcomes as rewards (more advanced)

---

## Part 4: Step-by-Step Implementation

### Step 1: Data Preparation Script

Convert trajectories to training format (ChatML/ShareGPT format)

### Step 2A: Finetune LLaMA (Recommended)

- Use LoRA/QLoRA for efficient training
- Can run on single GPU (16GB+)
- Open source, no API costs

### Step 2B: Finetune GPT-4o-mini (Alternative)

- Easier setup, just upload JSONL
- Higher cost (~$100-500 for 2000 examples)
- Good baseline

### Step 3: Simulated Human Agent

Use Claude to play the human role based on:

- Target personality type
- Hidden preferences
- Conversation history

### Step 4: Evaluation Framework

Run finetuned agent vs baseline and measure:

- Success rate
- Average turns to success
- Preference alignment accuracy
- Counterfactual quality

---

## Part 5: Recommended Approach

**Best Setup for Your Use Case:**

1. **Finetune LLaMA-3.1-8B** (open source, cost-effective)

   - Use QLoRA for memory efficiency
   - Train on single GPU (A100/4090)
   - ~2-4 hours training time

2. **Simulated Human: Claude Sonnet**

   - Already have API access
   - High quality responses
   - Follows personality traits well

3. **Evaluation: Head-to-head comparison**
   - Baseline (GPT-4 zero-shot)
   - Finetuned LLaMA
   - Measure on 100 held-out scenarios

---

## Part 6: Estimated Costs

**Option A: LLaMA Finetuning**

- Compute: $1-2/hour on RunPod/Lambda (A100)
- Training: ~3 hours = $3-6
- Inference: Free (run locally or $0.20/hour)
- **Total: ~$10-20**

**Option B: GPT-4o-mini Finetuning**

- Training: 2000 examples × 500 tokens avg = 1M tokens
- Cost: ~$8 per 1M tokens
- Inference: $0.15/1M input + $0.60/1M output
- Evaluation: 100 scenarios × 5 turns × 1K tokens = 500K tokens ≈ $3
- **Total: ~$50-100**

---

## Part 7: Quick Start Commands

### Setup Environment

```bash
# Create virtual environment
python3 -m venv ptom_env
source ptom_env/bin/activate

# Install packages
pip install torch transformers datasets peft trl accelerate bitsandbytes anthropic openai wandb

# Set API keys
export ANTHROPIC_API_KEY=sk-ant-xxx  # Get from console.anthropic.com
export HUGGINGFACE_TOKEN=hf_xxx  # Get from huggingface.co
export WANDB_API_KEY=xxx  # Optional, get from wandb.ai
```

### Workflow

```bash
# 1. Prepare training data
python prepare_training_data.py \
  --input business_vacation_traj.jsonl \
  --output training_data.jsonl \
  --format chatml

# 2. Finetune model
python finetune_llama.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --data training_data.jsonl \
  --output models/ptom-llama-8b

# 3. Run evaluation
python evaluate_agent.py \
  --model models/ptom-llama-8b \
  --human-agent claude-sonnet-4 \
  --scenarios test_scenarios.jsonl \
  --num-eval 100
```

---

## Part 8: Next Steps

I can create the following scripts for you:

1. **prepare_training_data.py** - Convert trajectories to training format
2. **finetune_llama.py** - QLoRA finetuning script
3. **finetune_gpt.py** - OpenAI finetuning script
4. **simulated_human.py** - Claude-based human agent
5. **evaluate_agent.py** - Full evaluation pipeline
6. **compare_models.py** - Benchmark multiple models

Would you like me to create these scripts?
