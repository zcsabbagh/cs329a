#!/usr/bin/env python3
"""
Evaluate Baseline Agent (No Finetuning)

This script evaluates a simple baseline agent that:
- Asks generic questions
- Makes random recommendations

Use this to establish baseline performance before finetuning.

Usage:
    python evaluate_baseline.py \
        --scenarios ../business_vacation_scenarios.jsonl \
        --num-scenarios 50 \
        --output results/baseline_evaluation.json
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm

from simulated_human import SimulatedHuman


class BaselineAgent:
    """
    Simple baseline agent that asks generic questions and makes random recommendations.

    This represents a naive approach without Planning ToM.
    """

    def __init__(self):
        self.generic_questions = [
            "What matters most to you in this decision?",
            "Can you tell me more about your priorities?",
            "What are your initial thoughts on the two options?",
            "Is there anything specific you're looking for?",
            "How do you feel about each option?",
            "What concerns do you have, if any?",
            "Have you thought about what's most important to you here?"
        ]
        self.turn_count = 0

    def generate_action(self, scenario: Dict, conversation_history: List[Dict], max_new_tokens=None) -> Dict:
        """
        Generate next action.

        Strategy:
        - Ask 4-5 generic questions
        - Then make a random recommendation
        """

        self.turn_count = len(conversation_history)

        # After 4-5 turns, make random recommendation
        if self.turn_count >= random.randint(4, 5):
            # Make random recommendation
            option = random.choice(["A", "B"])

            option_key = f"option_{option.lower()}"
            option_name = scenario.get(option_key, {}).get('name', f'Option {option}')

            reasoning = "Based on our conversation, this seems like a good fit for you."

            return {
                "reasoning": reasoning,
                "counterfactuals": {},
                "action_type": "ACT",
                "action": f"I recommend {option_name} (Option {option}). {reasoning}",
                "belief_state": {}
            }

        else:
            # Ask a generic question
            question_idx = self.turn_count % len(self.generic_questions)
            question = self.generic_questions[question_idx]

            return {
                "reasoning": "Gathering general information",
                "counterfactuals": {},
                "action_type": "ASK",
                "action": question,
                "belief_state": {}
            }


def load_scenarios(file_path: Path, num_scenarios: int = None) -> List[Dict]:
    """Load scenarios for evaluation."""

    scenarios = []
    with open(file_path, 'r') as f:
        for line in f:
            scenarios.append(json.loads(line))

    if num_scenarios:
        random.shuffle(scenarios)
        scenarios = scenarios[:num_scenarios]

    return scenarios


def run_conversation(
    agent: BaselineAgent,
    human: SimulatedHuman,
    scenario: Dict,
    max_turns: int = 10
) -> Dict:
    """
    Run a full conversation between baseline agent and human.

    Returns:
        {
            "scenario_id": str,
            "turns": List[Dict],
            "success": bool,
            "num_turns": int,
            "final_recommendation": str,
            "correct_option": str
        }
    """

    # Get target personality and preferences
    target_prefs = scenario.get('target_preferences', {})

    # Assign personality (cycle through types)
    personalities = [
        {"type": "luxury_focused", "description": "Values premium experiences", "response_style": "Emphasizes quality"},
        {"type": "budget_conscious", "description": "Careful about spending", "response_style": "Mentions costs"},
        {"type": "adventure_seeker", "description": "Thrives on challenges", "response_style": "Excited by experiences"},
        {"type": "culture_enthusiast", "description": "Passionate about learning", "response_style": "Asks about history"},
        {"type": "convenience_prioritizer", "description": "Values ease", "response_style": "Prefers simplicity"}
    ]

    scenario_id = scenario.get('scenario_id', '')
    idx = int(scenario_id.split('_')[-1]) if '_' in scenario_id else 0
    personality = personalities[idx % len(personalities)]

    conversation_history = []
    turns = []

    for turn_num in range(1, max_turns + 1):
        # Agent generates action
        agent_output = agent.generate_action(scenario, conversation_history)

        action_type = agent_output.get('action_type', 'ASK')
        action = agent_output.get('action', '')

        # Check if agent is making final recommendation
        if action_type.upper() == "ACT" or "recommend" in action.lower():
            # Extract recommendation
            if "option a" in action.lower():
                recommendation = "A"
            elif "option b" in action.lower():
                recommendation = "B"
            else:
                # Random if unclear
                recommendation = random.choice(["A", "B"])

            # Evaluate with human
            eval_result = human.evaluate_final_recommendation(
                scenario,
                target_prefs,
                recommendation,
                action
            )

            turns.append({
                "turn": turn_num,
                "agent_action": action,
                "action_type": action_type,
                "human_response": eval_result['response'],
                "final_turn": True
            })

            return {
                "scenario_id": scenario.get('scenario_id'),
                "turns": turns,
                "success": eval_result['persuader_goal_achieved'],
                "num_turns": turn_num,
                "final_recommendation": recommendation,
                "persuader_goal": eval_result['persuader_goal'],
                "naturally_preferred_option": eval_result['naturally_preferred_option'],
                "recommended_option": eval_result['recommended_option'],
                "target_accepted": eval_result['accepts'],
                "satisfaction": eval_result['satisfaction']
            }

        # Human responds
        human_response = human.respond(
            scenario,
            personality,
            target_prefs,
            conversation_history,
            action
        )

        # Record turn
        turns.append({
            "turn": turn_num,
            "agent_action": action,
            "action_type": action_type,
            "human_response": human_response,
            "final_turn": False
        })

        # Update history
        conversation_history.append({
            "action": action,
            "response": human_response
        })

    # Max turns reached without recommendation
    return {
        "scenario_id": scenario.get('scenario_id'),
        "turns": turns,
        "success": False,
        "num_turns": max_turns,
        "final_recommendation": None,
        "correct_option": "unknown",
        "satisfaction": 0.0,
        "failure_reason": "max_turns_exceeded"
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline Agent (No Finetuning)")

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
        default="results/baseline_evaluation.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Baseline Agent Evaluation (No Finetuning)")
    print("=" * 70)
    print("\nThis evaluates a simple baseline that:")
    print("  - Asks generic questions")
    print("  - Makes random recommendations")
    print("\nUse this to establish baseline performance.\n")

    # Initialize agents
    print("🤖 Initializing baseline agent...")
    agent = BaselineAgent()

    print("👤 Initializing simulated human (Claude)...")
    human = SimulatedHuman()

    # Load scenarios
    print(f"\n📂 Loading scenarios from {args.scenarios}")
    scenarios = load_scenarios(Path(args.scenarios), args.num_scenarios)
    print(f"✅ Loaded {len(scenarios)} scenarios")

    # Run evaluation
    print(f"\n🎯 Starting Evaluation")
    print("=" * 70)

    results = []

    for scenario in tqdm(scenarios, desc="Evaluating"):
        result = run_conversation(agent, human, scenario, args.max_turns)
        results.append(result)

    # Calculate metrics
    print("\n" + "=" * 70)
    print("📊 Results")
    print("=" * 70)

    successes = sum(1 for r in results if r['success'])
    success_rate = successes / len(results) * 100

    successful_results = [r for r in results if r['success']]
    avg_turns = sum(r['num_turns'] for r in successful_results) / len(successful_results) if successful_results else 0
    avg_satisfaction = sum(r['satisfaction'] for r in results) / len(results)

    print(f"\n🎲 Baseline Performance:")
    print(f"  • Success Rate: {success_rate:.1f}% ({successes}/{len(results)})")
    print(f"  • Average Turns (successful): {avg_turns:.2f}")
    print(f"  • Average Satisfaction: {avg_satisfaction:.2f}")

    print(f"\n💡 Interpretation:")
    if success_rate < 55:
        print(f"  Random baseline (~50% expected). Room for improvement with finetuning!")
    elif success_rate < 65:
        print(f"  Slightly better than random. Finetuning should help significantly.")
    else:
        print(f"  Surprisingly good! Finetuning might still improve further.")

    # Save results
    output_data = {
        "agent_type": "baseline_random",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "summary": {
            "num_scenarios": len(results),
            "success_rate": success_rate,
            "num_successes": successes,
            "avg_turns": avg_turns,
            "avg_satisfaction": avg_satisfaction
        },
        "results": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("✅ Baseline Evaluation Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review results in the JSON file")
    print("  2. Finetune a model (see ../finetuning/)")
    print("  3. Compare baseline vs finetuned performance")


if __name__ == "__main__":
    main()
