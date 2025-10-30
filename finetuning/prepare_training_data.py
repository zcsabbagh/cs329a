#!/usr/bin/env python3
"""
Convert Planning ToM trajectories to training format for LLaMA finetuning.

This script converts the conversation trajectories into the ChatML format
suitable for instruction finetuning.

Format:
- System: Scenario context + available facts + current task
- User: Conversation history
- Assistant: Internal reasoning + counterfactual analysis + action

Usage:
    python prepare_training_data.py \
        --input ../business_vacation_traj.jsonl \
        --output data/training_data.jsonl \
        --format chatml \
        --train-split 0.9
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def format_scenario_context(trajectory: Dict) -> str:
    """Format scenario information as context."""
    scenario = trajectory['scenario']
    condition = trajectory['condition']
    target_prefs = trajectory['target_preferences']

    # Get scenario details from available_facts (infer from context)
    context_parts = []

    context_parts.append(f"Scenario Type: {scenario}")
    context_parts.append(f"Information Condition: {condition}")

    # Target preferences
    if condition == "PARTIAL" and target_prefs.get('stated_explicitly'):
        revealed = target_prefs.get('partially_revealed_preferences', [])
        context_parts.append(f"Known Preferences: {', '.join(revealed)}")
        hidden = target_prefs.get('hidden_preferences', [])
        context_parts.append(f"Hidden Preferences: {', '.join(hidden)}")
    elif condition == "HIDDEN":
        context_parts.append("All preferences are hidden - must be inferred through conversation")

    # Available facts
    facts = trajectory.get('available_facts', [])
    if facts:
        context_parts.append(f"\nAvailable Facts ({len(facts)} total):")
        for fact in facts:
            context_parts.append(f"  [{fact['id']}] {fact['content']}")
            context_parts.append(f"      Favors: Option {fact['favors']}, Dimension: {fact['dimension']}")

    return "\n".join(context_parts)


def format_turn_for_training(
    trajectory: Dict,
    turn_data: Dict,
    conversation_history: List[Dict],
    format_type: str = "chatml"
) -> Dict:
    """
    Format a single turn into training format.

    Args:
        trajectory: Full trajectory data
        turn_data: Current turn data
        conversation_history: List of previous turns
        format_type: "chatml" or "sharegpt"

    Returns:
        Formatted training example
    """

    # Build system message with scenario context
    system_message = "You are an expert advocate using Planning Theory of Mind to help people make decisions.\n\n"
    system_message += "Your goal is to:\n"
    system_message += "1. Infer the target's hidden preferences through strategic questioning\n"
    system_message += "2. Maintain belief states about their desires and confidence\n"
    system_message += "3. Use counterfactual reasoning to evaluate different actions\n"
    system_message += "4. Recommend the option that best aligns with their true preferences\n\n"
    system_message += "SCENARIO:\n"
    system_message += format_scenario_context(trajectory)

    # Build user message with conversation history
    user_message = ""
    if len(conversation_history) == 0:
        user_message = "[Start of conversation - no history yet]\n\n"
        user_message += "Current situation: The target is considering their options. What should you do first?"
    else:
        user_message = "Conversation so far:\n\n"
        for prev_turn in conversation_history:
            user_message += f"You: {prev_turn['action']}\n"
            user_message += f"Target: {prev_turn['response']}\n\n"
        user_message += "What should you do next?"

    # Build assistant message with full reasoning
    assistant_message = ""

    # Add current belief state
    belief = turn_data.get('belief_state', {})
    if belief:
        assistant_message += "BELIEF STATE:\n"
        desires = belief.get('target_desires', {})
        if desires:
            assistant_message += "Target desires: " + ", ".join([f"{k}: {v:.2f}" for k, v in desires.items()]) + "\n"
        assistant_message += f"Confidence: {belief.get('confidence_level', 'unknown')}\n"
        assistant_message += f"Information entropy: {belief.get('information_entropy', 0):.2f}\n\n"

    # Add reasoning
    reasoning = turn_data.get('reasoning', '')
    if reasoning:
        assistant_message += f"REASONING:\n{reasoning}\n\n"

    # Add counterfactual analysis
    counterfactuals = turn_data.get('counterfactual_analysis', {})
    if counterfactuals:
        assistant_message += "COUNTERFACTUAL ANALYSIS:\n"
        for option_name, analysis in counterfactuals.items():
            p_success = analysis.get('p_success', 0)
            reasoning = analysis.get('reasoning', '')
            assistant_message += f"{option_name}: p_success={p_success:.2f}\n"
            assistant_message += f"  {reasoning}\n"
        assistant_message += "\n"

    # Add chosen action
    action_type = turn_data.get('action_type', '')
    action = turn_data.get('action', '')
    assistant_message += f"ACTION ({action_type}):\n{action}"

    # Format based on type
    if format_type == "chatml":
        return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
    elif format_type == "sharegpt":
        return {
            "conversations": [
                {"from": "system", "value": system_message},
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_message}
            ]
        }
    else:
        raise ValueError(f"Unknown format: {format_type}")


def convert_trajectory_to_training(
    trajectory: Dict,
    format_type: str = "chatml"
) -> List[Dict]:
    """Convert a full trajectory into multiple training examples (one per turn)."""

    training_examples = []
    conversation_history = []

    turns = trajectory.get('trajectory', [])

    for turn_idx, turn_data in enumerate(turns):
        # Create training example for this turn
        example = format_turn_for_training(
            trajectory,
            turn_data,
            conversation_history,
            format_type
        )

        # Add metadata
        example['game_id'] = trajectory.get('game_id', 'unknown')
        example['turn'] = turn_idx + 1
        example['strategy_type'] = trajectory.get('strategy_type', 'unknown')
        example['success'] = trajectory.get('success', False)

        training_examples.append(example)

        # Update conversation history for next turn
        conversation_history.append({
            'action': turn_data.get('action', ''),
            'response': turn_data.get('response', '')
        })

    return training_examples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Planning ToM trajectories for LLaMA finetuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to trajectories JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for training data"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="chatml",
        choices=["chatml", "sharegpt"],
        help="Output format (default: chatml)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)"
    )
    parser.add_argument(
        "--filter-strategy",
        type=str,
        default=None,
        choices=["optimal", "alternative_success", "failed", "recovery", "information_efficiency"],
        help="Only include specific strategy type (optional)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of training examples to generate (optional)"
    )

    args = parser.parse_args()

    # Load trajectories
    print(f"📂 Loading trajectories from {args.input}...")
    trajectories = []
    with open(args.input, 'r') as f:
        for line in f:
            try:
                traj = json.loads(line)

                # Filter by strategy if specified
                if args.filter_strategy and traj.get('strategy_type') != args.filter_strategy:
                    continue

                trajectories.append(traj)
            except:
                continue

    print(f"✅ Loaded {len(trajectories)} trajectories")

    # Convert to training examples
    print(f"\n🔄 Converting trajectories to {args.format} format...")
    all_examples = []

    for traj in trajectories:
        examples = convert_trajectory_to_training(traj, args.format)
        all_examples.extend(examples)

    print(f"✅ Generated {len(all_examples)} training examples")

    # Limit if specified
    if args.max_examples and len(all_examples) > args.max_examples:
        random.shuffle(all_examples)
        all_examples = all_examples[:args.max_examples]
        print(f"⚠️  Limited to {args.max_examples} examples")

    # Split train/val
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * args.train_split)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    print(f"\n📊 Split:")
    print(f"  • Training: {len(train_examples)} examples")
    print(f"  • Validation: {len(val_examples)} examples")

    # Write training data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n✅ Training data saved to: {output_path}")

    # Write validation data
    if val_examples:
        val_path = output_path.parent / output_path.name.replace('training', 'validation')
        with open(val_path, 'w') as f:
            for example in val_examples:
                f.write(json.dumps(example) + '\n')
        print(f"✅ Validation data saved to: {val_path}")

    # Print statistics
    print(f"\n📊 Statistics:")
    strategy_counts = defaultdict(int)
    success_counts = defaultdict(int)

    for example in all_examples:
        strategy_counts[example['strategy_type']] += 1
        if example['success']:
            success_counts[example['strategy_type']] += 1

    print(f"\nBy strategy type:")
    for strategy in sorted(strategy_counts.keys()):
        total = strategy_counts[strategy]
        successes = success_counts[strategy]
        print(f"  • {strategy}: {total} examples ({successes} successful, {successes/total*100:.1f}%)")

    print("\n" + "=" * 70)
    print("✅ Data preparation complete!")
    print("=" * 70)
    print("\nNext step:")
    print(f"  python finetune_llama.py --train-data {output_path}")


if __name__ == "__main__":
    main()
