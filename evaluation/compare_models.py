#!/usr/bin/env python3
"""
Compare Multiple Planning ToM Models

Runs multiple models (baseline and finetuned) on the same scenarios
and generates comparison statistics and visualizations.

Usage:
    python compare_models.py \
        --models baseline gpt-4o-mini ../finetuning/models/ptom-llama-8b \
        --scenarios ../business_vacation_scenarios.jsonl \
        --num-scenarios 50 \
        --output results/comparison.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate_agent import PlanningTomAgent, load_scenarios, run_conversation
from simulated_human import SimulatedHuman


def create_baseline_agent():
    """Create a simple baseline agent (random actions)."""

    class BaselineAgent:
        """Random baseline that asks generic questions."""

        def __init__(self):
            self.questions = [
                "What matters most to you in this decision?",
                "Can you tell me more about your priorities?",
                "What are your thoughts on the options?",
                "Is there anything specific you're looking for?",
                "How do you feel about each option?"
            ]
            self.question_idx = 0

        def generate_action(self, scenario, conversation_history, max_new_tokens=None):
            """Generate a generic question or random recommendation."""

            import random

            if len(conversation_history) >= 5:
                # Make random recommendation
                option = random.choice(["A", "B"])
                option_name = scenario.get(f'option_{option.lower()}', {}).get('name', f'Option {option}')

                return {
                    "reasoning": "Based on our conversation",
                    "counterfactuals": {},
                    "action_type": "ACT",
                    "action": f"I recommend {option_name} (Option {option}) as it seems to fit your needs."
                }
            else:
                # Ask generic question
                question = self.questions[self.question_idx % len(self.questions)]
                self.question_idx += 1

                return {
                    "reasoning": "Gathering information",
                    "counterfactuals": {},
                    "action_type": "ASK",
                    "action": question
                }

    return BaselineAgent()


def evaluate_model(
    model_name: str,
    model,
    human: SimulatedHuman,
    scenarios: List[Dict],
    max_turns: int = 10
) -> List[Dict]:
    """Evaluate a single model on all scenarios."""

    from tqdm import tqdm

    print(f"\n🤖 Evaluating: {model_name}")
    print("-" * 70)

    results = []
    for scenario in tqdm(scenarios, desc=f"  {model_name}"):
        result = run_conversation(model, human, scenario, max_turns)
        result['model'] = model_name
        results.append(result)

    # Calculate metrics
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / len(results) * 100
    successful_results = [r for r in results if r['success']]
    avg_turns = sum(r['num_turns'] for r in successful_results) / len(successful_results) if successful_results else 0

    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Avg Turns: {avg_turns:.2f}")

    return results


def create_comparison_plots(all_results: List[Dict], output_dir: Path):
    """Create visualization plots comparing models."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    df = pd.DataFrame(all_results)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # Plot 1: Success Rate by Model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    success_by_model = df.groupby('model')['success'].agg(['mean', 'count'])
    success_by_model['mean'] *= 100  # Convert to percentage

    success_by_model['mean'].plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_title('Success Rate by Model', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(success_by_model['mean']):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    # Plot 2: Average Turns (successful conversations only)
    successful_df = df[df['success'] == True]
    avg_turns = successful_df.groupby('model')['num_turns'].mean()

    avg_turns.plot(kind='bar', ax=ax2, color='coral')
    ax2.set_title('Average Turns to Success', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Turns', fontsize=12)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, v in enumerate(avg_turns):
        ax2.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: {output_dir / 'model_comparison.png'}")

    # Plot 3: Success Rate Distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    models = df['model'].unique()
    for model in models:
        model_df = df[df['model'] == model]
        success_rates = model_df.groupby('scenario_id')['success'].mean()
        ax.hist(success_rates * 100, alpha=0.5, label=model, bins=10)

    ax.set_title('Distribution of Success Rates', fontsize=14, fontweight='bold')
    ax.set_xlabel('Success Rate (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'success_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: {output_dir / 'success_distribution.png'}")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Compare Planning ToM Models")

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of models ('baseline', 'gpt-4o-mini', or path to finetuned model)"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        required=True,
        help="Path to scenarios JSONL"
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=50,
        help="Number of scenarios to evaluate"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns per conversation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model for finetuned models"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Model Comparison Evaluation")
    print("=" * 70)

    # Load scenarios (same scenarios for all models)
    print(f"\n📂 Loading scenarios...")
    scenarios = load_scenarios(Path(args.scenarios), args.num_scenarios)
    print(f"✅ Loaded {len(scenarios)} scenarios")

    # Initialize simulated human
    human = SimulatedHuman()

    # Evaluate each model
    all_results = []

    for model_name in args.models:
        if model_name.lower() == "baseline":
            model = create_baseline_agent()
            name = "Random Baseline"
        elif model_name.lower().startswith("gpt"):
            print(f"\n⚠️  GPT models not yet implemented. Use baseline or finetuned LLaMA.")
            continue
        else:
            # Finetuned model path
            model = PlanningTomAgent(model_name, args.base_model)
            name = Path(model_name).name

        results = evaluate_model(name, model, human, scenarios, args.max_turns)
        all_results.extend(results)

    # Calculate comparison statistics
    print("\n" + "=" * 70)
    print("📊 Comparison Summary")
    print("=" * 70)

    df = pd.DataFrame(all_results)

    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        successes = model_df['success'].sum()
        total = len(model_df)
        success_rate = successes / total * 100

        successful_df = model_df[model_df['success'] == True]
        avg_turns = successful_df['num_turns'].mean() if len(successful_df) > 0 else 0
        avg_satisfaction = model_df['satisfaction'].mean()

        print(f"\n{model_name}:")
        print(f"  Success Rate: {success_rate:.1f}% ({successes}/{total})")
        print(f"  Avg Turns: {avg_turns:.2f}")
        print(f"  Avg Satisfaction: {avg_satisfaction:.2f}")

    # Save results
    output_data = {
        "models": args.models,
        "num_scenarios": len(scenarios),
        "config": vars(args),
        "results": all_results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    # Create plots
    print(f"\n📊 Creating comparison plots...")
    create_comparison_plots(all_results, output_path.parent)

    print("\n" + "=" * 70)
    print("✅ Comparison Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
