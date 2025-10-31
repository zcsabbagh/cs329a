#!/usr/bin/env python3
"""
Fair Comparison: Baseline vs. Fine-tuned Model

This script runs a controlled experiment comparing:
- Baseline model (random questioning + random recommendations)
- Fine-tuned model (Ira's Llama-3.1-8B model)

Both models:
- See the SAME 50 scenarios (controlled)
- Use the SAME SimulatedHuman target (Claude)
- Are evaluated with the SAME corrected persuasion logic

Usage:
    python run_fair_comparison.py
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Import evaluation modules
import sys
sys.path.append(str(Path(__file__).parent / "evaluation"))

from evaluation.evaluate_baseline import BaselineAgent, run_conversation as run_baseline_conversation
from evaluation.evaluate_together_agent import TogetherAgent, run_conversation as run_together_conversation
from evaluation.simulated_human import SimulatedHuman


def load_api_keys():
    """Load API keys from environment or .env file."""
    # Try to load from .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    together_key = os.environ.get("TOGETHER_API_KEY")
    
    if not anthropic_key or not together_key:
        raise ValueError("Missing API keys. Ensure ANTHROPIC_API_KEY and TOGETHER_API_KEY are set.")
    
    return anthropic_key, together_key


def load_scenarios(file_path: Path, num_scenarios: int, seed: int = 42) -> List[Dict]:
    """Load and select scenarios for evaluation."""
    scenarios = []
    with open(file_path, 'r') as f:
        for line in f:
            scenarios.append(json.loads(line))
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Randomly select scenarios
    if num_scenarios < len(scenarios):
        scenarios = random.sample(scenarios, num_scenarios)
    else:
        scenarios = scenarios[:num_scenarios]
    
    return scenarios


def evaluate_baseline(scenarios: List[Dict], anthropic_key: str, max_turns: int = 10) -> Dict:
    """Evaluate baseline agent."""
    print("\n" + "=" * 70)
    print("🎲 BASELINE MODEL EVALUATION")
    print("=" * 70)
    
    agent = BaselineAgent()
    human = SimulatedHuman()
    
    # Define personalities (same as other evaluations)
    personalities = [
        {"type": "luxury_focused", "description": "Values premium experiences", "response_style": "Emphasizes quality"},
        {"type": "budget_conscious", "description": "Careful about spending", "response_style": "Mentions costs"},
        {"type": "adventure_seeker", "description": "Thrives on challenges", "response_style": "Excited by experiences"},
        {"type": "culture_enthusiast", "description": "Passionate about learning", "response_style": "Asks about history"},
        {"type": "convenience_prioritizer", "description": "Values ease", "response_style": "Prefers simplicity"}
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] Evaluating scenario {scenario.get('scenario_id')}...", end="")
        
        # Assign personality based on scenario ID
        scenario_id = scenario.get('scenario_id', '')
        idx = int(scenario_id.split('_')[-1]) if '_' in scenario_id else 0
        personality = personalities[idx % len(personalities)]
        
        # Get target preferences
        target_prefs = scenario.get('target_preferences', {})
        
        # Run conversation
        result = run_baseline_conversation(agent, human, scenario, max_turns)
        
        results.append(result)
        
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f" {status}")
    
    # Calculate statistics
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / len(results) * 100 if results else 0
    
    successful_results = [r for r in results if r['success']]
    avg_turns = sum(r['num_turns'] for r in successful_results) / len(successful_results) if successful_results else 0
    avg_satisfaction = sum(r['satisfaction'] for r in results) / len(results) if results else 0
    
    print(f"\n📊 Baseline Results:")
    print(f"  • Success Rate: {success_rate:.1f}% ({successes}/{len(results)})")
    print(f"  • Avg Turns (successful): {avg_turns:.2f}")
    print(f"  • Avg Satisfaction: {avg_satisfaction:.2f}")
    
    return {
        "agent_type": "baseline_random",
        "results": results,
        "statistics": {
            "success_rate": success_rate / 100,
            "num_successes": successes,
            "num_total": len(results),
            "avg_turns": avg_turns,
            "avg_satisfaction": avg_satisfaction
        }
    }


def evaluate_finetuned(scenarios: List[Dict], together_key: str, anthropic_key: str, 
                       model_id: str, max_turns: int = 8) -> Dict:
    """Evaluate fine-tuned model."""
    print("\n" + "=" * 70)
    print("🤖 FINE-TUNED MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: {model_id}")
    
    agent = TogetherAgent(model_id, together_key)
    human = SimulatedHuman()
    
    # Define personalities (same as baseline)
    personalities = [
        {"type": "luxury_focused", "description": "Values premium experiences", "response_style": "Emphasizes quality"},
        {"type": "budget_conscious", "description": "Careful about spending", "response_style": "Mentions costs"},
        {"type": "adventure_seeker", "description": "Thrives on challenges", "response_style": "Excited by experiences"},
        {"type": "culture_enthusiast", "description": "Passionate about learning", "response_style": "Asks about history"},
        {"type": "convenience_prioritizer", "description": "Values ease", "response_style": "Prefers simplicity"}
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] Evaluating scenario {scenario.get('scenario_id')}...", end="")
        
        # Assign personality based on scenario ID
        scenario_id = scenario.get('scenario_id', '')
        idx = int(scenario_id.split('_')[-1]) if '_' in scenario_id else 0
        personality = personalities[idx % len(personalities)]
        
        # Get target preferences
        target_prefs = scenario.get('target_preferences', {})
        
        # Run conversation
        result = run_together_conversation(
            agent, human, scenario, personality, target_prefs, max_turns
        )
        
        results.append({
            "scenario_id": scenario.get('scenario_id'),
            "success": result.get('success', False),
            "turns": result.get('turns', 0),
            "satisfaction": result.get('satisfaction', 0.0),
            "persuader_goal": result.get('persuader_goal', 'A'),
            "naturally_preferred_option": result.get('naturally_preferred_option', 'unknown'),
            "recommended_option": result.get('recommended_option', 'unknown'),
            "target_accepted": result.get('target_accepted', False),
            "conversation": result.get('conversation', [])
        })
        
        status = "✓ SUCCESS" if result.get('success', False) else "✗ FAILED"
        print(f" {status}")
    
    # Calculate statistics
    successes = [r for r in results if r['success']]
    success_rate = len(successes) / len(results) if results else 0
    
    avg_turns = sum(r['turns'] for r in successes) / len(successes) if successes else 0
    avg_satisfaction = sum(r['satisfaction'] for r in results) / len(results) if results else 0
    
    print(f"\n📊 Fine-tuned Results:")
    print(f"  • Success Rate: {success_rate*100:.1f}% ({len(successes)}/{len(results)})")
    print(f"  • Avg Turns (successful): {avg_turns:.2f}")
    print(f"  • Avg Satisfaction: {avg_satisfaction:.2f}")
    
    return {
        "agent_type": "finetuned",
        "model_id": model_id,
        "results": results,
        "statistics": {
            "success_rate": success_rate,
            "num_successes": len(successes),
            "num_total": len(results),
            "avg_turns": avg_turns,
            "avg_satisfaction": avg_satisfaction
        }
    }


def compare_results(baseline_stats: Dict, finetuned_stats: Dict) -> Dict:
    """Compare baseline and fine-tuned results."""
    print("\n" + "=" * 70)
    print("📈 COMPARISON SUMMARY")
    print("=" * 70)
    
    baseline_sr = baseline_stats['success_rate'] * 100
    finetuned_sr = finetuned_stats['success_rate'] * 100
    improvement = finetuned_sr - baseline_sr
    
    print(f"\n✨ Success Rate:")
    print(f"  • Baseline:    {baseline_sr:.1f}%")
    print(f"  • Fine-tuned:  {finetuned_sr:.1f}%")
    print(f"  • Improvement: {improvement:+.1f}% ({'↑' if improvement > 0 else '↓'})")
    
    print(f"\n🎯 Average Turns (when successful):")
    print(f"  • Baseline:    {baseline_stats['avg_turns']:.2f}")
    print(f"  • Fine-tuned:  {finetuned_stats['avg_turns']:.2f}")
    
    print(f"\n😊 Average Satisfaction:")
    print(f"  • Baseline:    {baseline_stats['avg_satisfaction']:.2f}")
    print(f"  • Fine-tuned:  {finetuned_stats['avg_satisfaction']:.2f}")
    
    # Interpret results
    print(f"\n💡 Interpretation:")
    if improvement > 10:
        print(f"  ✓ Strong improvement! Fine-tuning significantly enhanced PToM abilities.")
    elif improvement > 5:
        print(f"  ✓ Moderate improvement. Fine-tuning helped, but there's room for more.")
    elif improvement > 0:
        print(f"  → Slight improvement. Consider additional training or prompt engineering.")
    else:
        print(f"  ⚠ No improvement. May need to revisit training data or model architecture.")
    
    return {
        "success_rate_improvement": improvement,
        "baseline_success_rate": baseline_sr,
        "finetuned_success_rate": finetuned_sr
    }


def main():
    parser = argparse.ArgumentParser(description="Fair comparison between baseline and fine-tuned models")
    
    parser.add_argument(
        "--scenarios",
        type=str,
        default="business_vacation_scenarios.jsonl",
        help="Path to scenarios JSONL file"
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=50,
        help="Number of scenarios to evaluate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scenario selection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="irawadee_5d65/Meta-Llama-3.1-8B-Instruct-Reference-ptom-agent-5713b704",
        help="Together AI fine-tuned model ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load API keys
    print("🔑 Loading API keys...")
    try:
        anthropic_key, together_key = load_api_keys()
        print("  ✓ API keys loaded")
    except ValueError as e:
        print(f"  ✗ Error: {e}")
        return
    
    # Load scenarios
    print(f"\n📂 Loading scenarios from {args.scenarios}...")
    scenarios_path = Path(args.scenarios)
    if not scenarios_path.exists():
        print(f"  ✗ Error: Scenarios file not found at {scenarios_path}")
        return
    
    scenarios = load_scenarios(scenarios_path, args.num_scenarios, args.seed)
    print(f"  ✓ Loaded {len(scenarios)} scenarios (seed={args.seed})")
    
    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evaluate baseline
    baseline_results = evaluate_baseline(scenarios, anthropic_key, max_turns=10)
    
    # Save baseline results
    baseline_path = output_dir / f"baseline_fair_comparison_{timestamp}.json"
    with open(baseline_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_scenarios": args.num_scenarios,
                "seed": args.seed,
                "scenarios_file": args.scenarios
            },
            **baseline_results
        }, f, indent=2)
    print(f"\n💾 Baseline results saved to: {baseline_path}")
    
    # Evaluate fine-tuned model
    finetuned_results = evaluate_finetuned(
        scenarios, together_key, anthropic_key, args.model, max_turns=8
    )
    
    # Save fine-tuned results
    finetuned_path = output_dir / f"finetuned_fair_comparison_{timestamp}.json"
    with open(finetuned_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_scenarios": args.num_scenarios,
                "seed": args.seed,
                "scenarios_file": args.scenarios,
                "model": args.model
            },
            **finetuned_results
        }, f, indent=2)
    print(f"💾 Fine-tuned results saved to: {finetuned_path}")
    
    # Compare results
    comparison = compare_results(
        baseline_results['statistics'],
        finetuned_results['statistics']
    )
    
    # Save comparison
    comparison_path = output_dir / f"comparison_{timestamp}.json"
    with open(comparison_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "baseline_file": str(baseline_path),
            "finetuned_file": str(finetuned_path),
            "baseline_statistics": baseline_results['statistics'],
            "finetuned_statistics": finetuned_results['statistics'],
            "comparison": comparison
        }, f, indent=2)
    print(f"💾 Comparison saved to: {comparison_path}")
    
    print("\n" + "=" * 70)
    print("✅ FAIR COMPARISON COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print(f"  1. Review detailed results in {output_dir}/")
    print(f"  2. Analyze trajectories to understand model behavior")
    print(f"  3. If fine-tuned model underperforms, consider:")
    print(f"     - Reviewing training data quality")
    print(f"     - Adjusting fine-tuning hyperparameters")
    print(f"     - Adding more diverse training examples")


if __name__ == "__main__":
    main()

