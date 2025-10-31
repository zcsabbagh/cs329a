# Fair Baseline vs. Fine-tuned Model Comparison

This guide explains how to run a **statistically fair comparison** between your baseline and fine-tuned models for the PToM persuasion task.

## What This Does

✓ **Runs 50 scenarios** for baseline model  
✓ **Runs 50 scenarios** for fine-tuned model  
✓ **Uses the SAME scenarios** for both (controlled comparison)  
✓ **Same target agent** (SimulatedHuman with Claude)  
✓ **Same evaluation metrics** (corrected persuasion logic)  
✓ **Automatic comparison** with statistical summary

## Quick Start

```bash
# Make sure you're in the project directory
cd /Users/riyakarumanchi/Desktop/cs329a

# Run the fair comparison (uses API keys from .env)
python run_fair_comparison.py
```

That's it! The script will:
1. Load 50 random scenarios (same seed = reproducible)
2. Run baseline evaluation (~10-15 minutes)
3. Run fine-tuned evaluation (~10-15 minutes)
4. Save all results with timestamps
5. Print comparison summary

## Expected Output

```
🔑 Loading API keys...
  ✓ API keys loaded

📂 Loading scenarios...
  ✓ Loaded 50 scenarios (seed=42)

==================================================================
🎲 BASELINE MODEL EVALUATION
==================================================================
  [1/50] Evaluating scenario scenario_218... ✓ SUCCESS
  [2/50] Evaluating scenario scenario_248... ✗ FAILED
  ...

📊 Baseline Results:
  • Success Rate: 52.0% (26/50)
  • Avg Turns (successful): 6.2
  • Avg Satisfaction: 0.45

==================================================================
🤖 FINE-TUNED MODEL EVALUATION
==================================================================
Model: irawadee_5d65/Meta-Llama-3.1-8B-Instruct-Reference-ptom-agent-5713b704
  [1/50] Evaluating scenario scenario_218... ✓ SUCCESS
  [2/50] Evaluating scenario scenario_248... ✓ SUCCESS
  ...

📊 Fine-tuned Results:
  • Success Rate: 68.0% (34/50)
  • Avg Turns (successful): 5.8
  • Avg Satisfaction: 0.62

==================================================================
📈 COMPARISON SUMMARY
==================================================================

✨ Success Rate:
  • Baseline:    52.0%
  • Fine-tuned:  68.0%
  • Improvement: +16.0% (↑)

🎯 Average Turns (when successful):
  • Baseline:    6.20
  • Fine-tuned:  5.80

😊 Average Satisfaction:
  • Baseline:    0.45
  • Fine-tuned:  0.62

💡 Interpretation:
  ✓ Strong improvement! Fine-tuning significantly enhanced PToM abilities.
```

## Output Files

All results are saved in `evaluation/results/` with timestamps:

- **`baseline_fair_comparison_YYYYMMDD_HHMMSS.json`**  
  Full baseline results with all trajectories

- **`finetuned_fair_comparison_YYYYMMDD_HHMMSS.json`**  
  Full fine-tuned results with all trajectories

- **`comparison_YYYYMMDD_HHMMSS.json`**  
  Statistical comparison summary

## Advanced Options

### Change number of scenarios
```bash
python run_fair_comparison.py --num-scenarios 100
```

### Use different model
```bash
python run_fair_comparison.py --model "your-model-id"
```

### Change random seed (for different scenario selection)
```bash
python run_fair_comparison.py --seed 123
```

### Full options
```bash
python run_fair_comparison.py \
    --scenarios business_vacation_scenarios.jsonl \
    --num-scenarios 50 \
    --seed 42 \
    --model "irawadee_5d65/Meta-Llama-3.1-8B-Instruct-Reference-ptom-agent-5713b704" \
    --output-dir evaluation/results
```

## What Makes This Fair?

1. **Same Scenarios**: Both models see identical payoff matrices, target preferences, and facts
2. **Same Target**: Both interact with the same SimulatedHuman (Claude) configuration
3. **Same Metrics**: Both use the corrected `persuader_goal_achieved` logic
4. **Random Seed**: Reproducible scenario selection (seed=42 by default)
5. **Equal Sample Size**: 50 trajectories each

This follows best practices from Anthropic research for fair model comparison.

## Interpreting Results

### Success Rate
- **> +15% improvement**: Strong evidence of PToM learning
- **+5-15% improvement**: Moderate improvement, consider more training
- **< +5% improvement**: May need to revisit training data or approach
- **Negative improvement**: Model may have overfit or training data issues

### Average Turns
- **Lower is better** (more efficient persuasion)
- Fine-tuned should use fewer turns if it's truly strategic

### Satisfaction
- **Higher is better** (target feels more respected/understood)
- Shows quality of interaction, not just success

## Troubleshooting

**API Key Errors?**
```bash
# Check your .env file
cat .env

# Should contain:
# ANTHROPIC_API_KEY=sk-ant-...
# TOGETHER_API_KEY=tgp_v1_...
```

**Import Errors?**
```bash
# Make sure you're in the right directory
pwd  # Should show: /Users/riyakarumanchi/Desktop/cs329a

# Install dependencies if needed
pip install anthropic together tqdm
```

**Too Slow?**
- Use `--num-scenarios 10` for quick test
- Full 50 scenarios takes ~20-30 minutes total

## Next Steps After Comparison

1. **Analyze Trajectories**  
   Open the JSON files and read example conversations to understand:
   - Why does fine-tuned succeed where baseline fails?
   - Is fine-tuned asking better questions?
   - Is it revealing facts more strategically?

2. **Check for Overfitting**  
   If fine-tuned succeeds on training scenarios but not test scenarios, you may have overfit.

3. **Prepare for Anthropic Application**  
   Include these results in your application:
   - Success rate improvement
   - Example trajectories showing strategic reasoning
   - Comparison to baseline

4. **Test on Actual MindGames Benchmark** (optional)  
   After this, you can adapt your agent to run on the official MindGames task for publication-quality results.

---

**Questions?** Review the code in `run_fair_comparison.py` — it's extensively commented!

