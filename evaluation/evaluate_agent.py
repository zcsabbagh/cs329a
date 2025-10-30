#!/usr/bin/env python3
"""
Evaluate Planning ToM Agent against Simulated Human

This script runs a finetuned model through complete conversations
with the simulated human and measures performance.

Usage:
    python evaluate_agent.py \
        --model ../finetuning/models/ptom-llama-8b \
        --scenarios ../business_vacation_scenarios.jsonl \
        --num-scenarios 100 \
        --output results/evaluation.json
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from simulated_human import SimulatedHuman


class PlanningTomAgent:
    """Wrapper for finetuned Planning ToM agent."""

    def __init__(
        self,
        model_path: str,
        base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str = "auto"
    ):
        """
        Load finetuned model.

        Args:
            model_path: Path to finetuned model (LoRA adapter)
            base_model: Base model name
            device: Device to load on
        """
        print(f"📥 Loading model from {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # Load adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            model_path
        )

        self.model.eval()
        print("✅ Model loaded")

    def generate_action(
        self,
        scenario: Dict,
        conversation_history: List[Dict],
        max_new_tokens: int = 1024
    ) -> Dict:
        """
        Generate next action (ASK, DISCLOSE, or ACT).

        Returns:
            {
                "reasoning": str,
                "counterfactuals": Dict,
                "action_type": str,
                "action": str,
                "belief_state": Dict
            }
        """

        # Format prompt (same as training format)
        system_msg = self._build_system_message(scenario)
        user_msg = self._build_user_message(conversation_history)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        assistant_response = generated.split("assistant")[-1].strip()

        # Parse response
        parsed = self._parse_response(assistant_response)

        return parsed

    def _build_system_message(self, scenario: Dict) -> str:
        """Build system message with scenario context."""

        msg = "You are an expert advocate using Planning Theory of Mind to help people make decisions.\n\n"
        msg += "Your goal is to:\n"
        msg += "1. Infer the target's hidden preferences through strategic questioning\n"
        msg += "2. Maintain belief states about their desires and confidence\n"
        msg += "3. Use counterfactual reasoning to evaluate different actions\n"
        msg += "4. Recommend the option that best aligns with their true preferences\n\n"

        msg += "SCENARIO:\n"
        msg += f"Type: {scenario.get('scenario', 'unknown')}\n"
        msg += f"Condition: {scenario.get('condition', 'HIDDEN')}\n\n"

        # Add available facts
        facts = scenario.get('available_facts', [])
        if facts:
            msg += f"Available Facts ({len(facts)} total):\n"
            for fact in facts[:10]:  # Limit to avoid context overflow
                msg += f"  [{fact['id']}] {fact['content']}\n"
                msg += f"      Favors: Option {fact['favors']}, Dimension: {fact['dimension']}\n"

        return msg

    def _build_user_message(self, conversation_history: List[Dict]) -> str:
        """Build user message with conversation history."""

        if not conversation_history:
            return "[Start of conversation]\n\nWhat should you do first?"

        msg = "Conversation so far:\n\n"
        for turn in conversation_history:
            msg += f"You: {turn['action']}\n"
            msg += f"Target: {turn['response']}\n\n"

        msg += "What should you do next?"
        return msg

    def _parse_response(self, response: str) -> Dict:
        """Parse model response into structured format."""

        # Extract sections
        result = {
            "raw_response": response,
            "reasoning": "",
            "counterfactuals": {},
            "action_type": "ASK",
            "action": "",
            "belief_state": {}
        }

        # Simple parsing (can be improved with more robust extraction)
        lines = response.split('\n')

        current_section = None
        for line in lines:
            line_lower = line.lower()

            if "reasoning:" in line_lower:
                current_section = "reasoning"
            elif "counterfactual" in line_lower:
                current_section = "counterfactuals"
            elif "action" in line_lower and ":" in line:
                current_section = "action"
                # Extract action type from line like "ACTION (ASK):"
                if "(" in line and ")" in line:
                    action_type = line.split("(")[1].split(")")[0].strip()
                    result["action_type"] = action_type
            elif current_section == "reasoning":
                result["reasoning"] += line + " "
            elif current_section == "action":
                result["action"] += line + " "

        result["reasoning"] = result["reasoning"].strip()
        result["action"] = result["action"].strip()

        return result


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
    agent: PlanningTomAgent,
    human: SimulatedHuman,
    scenario: Dict,
    max_turns: int = 10
) -> Dict:
    """
    Run a full conversation between agent and human.

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
        {"type": "luxury_focused", "description": "Values premium experiences", "response_style": "Emphasizes quality and comfort"},
        {"type": "budget_conscious", "description": "Careful about spending", "response_style": "Mentions costs and value"},
        {"type": "adventure_seeker", "description": "Thrives on challenges", "response_style": "Excited by new experiences"},
        {"type": "culture_enthusiast", "description": "Passionate about learning", "response_style": "Asks about history and arts"},
        {"type": "convenience_prioritizer", "description": "Values ease and efficiency", "response_style": "Prefers simplicity"}
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
                recommendation = "A"  # Default

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
                "agent_reasoning": agent_output.get('reasoning', ''),
                "human_response": eval_result['response'],
                "final_turn": True
            })

            return {
                "scenario_id": scenario.get('scenario_id'),
                "turns": turns,
                "success": eval_result['accepts'],
                "num_turns": turn_num,
                "final_recommendation": recommendation,
                "correct_option": eval_result['correct_option'],
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
            "agent_reasoning": agent_output.get('reasoning', ''),
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
    parser = argparse.ArgumentParser(description="Evaluate Planning ToM Agent")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to finetuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model name"
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
        default=100,
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
        default="results/evaluation.json",
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
    torch.manual_seed(args.seed)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load models
    print("=" * 70)
    print("Initializing Evaluation")
    print("=" * 70)

    agent = PlanningTomAgent(args.model, args.base_model)
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

    print(f"Success Rate: {success_rate:.1f}% ({successes}/{len(results)})")
    print(f"Average Turns (successful): {avg_turns:.2f}")
    print(f"Average Satisfaction: {avg_satisfaction:.2f}")

    # Save results
    output_data = {
        "model": args.model,
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
    print("=" * 70)


if __name__ == "__main__":
    main()
