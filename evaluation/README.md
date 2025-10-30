# Evaluation Pipeline for Planning ToM Agents

This directory contains scripts for evaluating finetuned persuasion agents against simulated humans.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY=sk-ant-xxx  # For simulated human
export OPENAI_API_KEY=sk-xxx  # Optional: for baseline comparison
export HUGGINGFACE_TOKEN=hf_xxx  # For loading LLaMA models
```

## Quick Start

### Step 1: Test Simulated Human
```bash
python simulated_human.py --test
```

This will run a quick test to verify the simulated human agent works correctly.

### Step 2: Evaluate Finetuned Model
```bash
python evaluate_agent.py \
  --model ../finetuning/models/ptom-llama-8b \
  --scenarios ../business_vacation_scenarios.jsonl \
  --num-scenarios 100 \
  --output results/ptom_llama_evaluation.json
```

### Step 3: Compare Models
```bash
python compare_models.py \
  --models baseline gpt-4o-mini ../finetuning/models/ptom-llama-8b \
  --scenarios ../business_vacation_scenarios.jsonl \
  --num-scenarios 50 \
  --output results/model_comparison.json
```

This will generate:
- Detailed results JSON
- Summary statistics
- Comparison plots

## Evaluation Metrics

The evaluation measures:

1. **Success Rate**: % of conversations where agent recommends the correct option
2. **Average Turns**: Mean number of turns to reach recommendation
3. **Preference Accuracy**: How well agent infers hidden preferences
4. **Belief Calibration**: Quality of confidence estimates
5. **Counterfactual Quality**: Whether agent considers alternatives appropriately

## Simulated Human

The simulated human is powered by Claude and:
- Follows the target personality type (e.g., budget_conscious, adventure_seeker)
- Has hidden preferences with specific strengths
- Responds naturally based on conversation history
- Reveals information strategically based on condition (HIDDEN/PARTIAL)

## Output Files

Results are saved in `results/` directory:
- `{model_name}_evaluation.json` - Detailed per-scenario results
- `{model_name}_summary.json` - Aggregate statistics
- `comparison_plot.png` - Visual comparison of models
- `success_rate_by_strategy.png` - Performance breakdown

## Model Types

The evaluation supports:

1. **Finetuned LLaMA** - Your trained model
2. **GPT-4/4o-mini** - OpenAI baseline (requires API key)
3. **Zero-shot Claude** - Anthropic baseline
4. **Random Agent** - Sanity check baseline

## Next Steps

After evaluation:
1. Review `results/` directory for detailed analysis
2. Identify failure modes from low-success scenarios
3. Iterate on training data or hyperparameters
4. Test on additional scenarios
