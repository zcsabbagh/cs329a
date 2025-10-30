#!/usr/bin/env python3
"""
Evaluate a Together AI finetuned agent against simulated humans.

Usage:
    python evaluate_together_agent.py \
        --model irawadee_5d65/Meta-Llama-3.1-8B-Instruct-Reference-ptom-agent-5713b704 \
        --scenarios ../business_vacation_scenarios.jsonl \
        --num-scenarios 50 \
        --output results/together_evaluation.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import re
from tqdm import tqdm
from together import Together
from simulated_human import SimulatedHuman


class TogetherAgent:
    """Agent using finetuned Together AI model."""

    def __init__(self, model_id: str, api_key: str):
        self.model_id = model_id
        self.client = Together(api_key=api_key)
        self.turn_count = 0
        self.max_turns = 8

    def generate_action(self, scenario: Dict, conversation_history: List[Dict]) -> Dict:
        """Generate next action using finetuned model."""

        # Build the prompt in the same format as training data
        system_message = self._build_system_message(scenario)
        user_message = self._build_user_message(conversation_history)

        # Call Together AI
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        # Extract text from response
        response_text = response.choices[0].message.content

        # Parse the response
        action = self._parse_response(response_text)

        self.turn_count += 1

        return action

    def _build_system_message(self, scenario: Dict) -> str:
        """Build system message matching training format."""

        # Format facts
        facts = scenario.get('available_facts', [])
        facts_text = ""
        for i, fact in enumerate(facts, 1):
            content = fact.get('content', fact.get('text', ''))
            facts_text += f"  [fact_{i}] {content}\n"
            facts_text += f"      Favors: {fact.get('favors', 'Option neutral')}, Dimension: {fact.get('dimension', 'neutral')}\n"

        system_prompt = f"""You are an expert advocate using Planning Theory of Mind to help people make decisions.

Your goal is to:
1. Infer the target's hidden preferences through strategic questioning
2. Maintain belief states about their desires and confidence
3. Use counterfactual reasoning to evaluate different actions
4. Recommend the option that best aligns with their true preferences

SCENARIO:
Scenario Type: {scenario.get('scenario_type', 'unknown')}
Information Condition: {scenario.get('condition', 'PARTIAL')}

Available Facts ({len(facts)} total):
{facts_text}"""

        return system_prompt

    def _build_user_message(self, conversation_history: List[Dict]) -> str:
        """Build user message with conversation history."""

        if not conversation_history:
            return "What should you do first?"

        # Format conversation
        conv_text = "Conversation so far:\n\n"
        for turn in conversation_history:
            conv_text += f"You: {turn['advocate']}\n"
            conv_text += f"Target: {turn['human']}\n\n"

        conv_text += "What should you do next?"

        return conv_text

    def _parse_response(self, response_text: str) -> Dict:
        """Parse model response to extract action."""

        # Look for ACTION section
        action_match = re.search(r'ACTION[:\s]+\(?(ASK|DISCLOSE|ACT)\)?[:\s]*(.*?)(?:\n\n|$)',
                                response_text, re.DOTALL | re.IGNORECASE)

        if action_match:
            action_type = action_match.group(1).upper()
            action_text = action_match.group(2).strip()

            return {
                "action_type": action_type,
                "action": action_text
            }

        # Fallback: look for recommendation pattern
        if re.search(r'recommend|suggest', response_text, re.IGNORECASE):
            return {
                "action_type": "ACT",
                "action": response_text.strip()
            }

        # Default: treat as ASK
        return {
            "action_type": "ASK",
            "action": response_text.strip()
        }

    def reset(self):
        """Reset turn counter."""
        self.turn_count = 0


def run_conversation(
    agent: TogetherAgent,
    human: SimulatedHuman,
    scenario: Dict,
    personality: Dict,
    target_preferences: Dict,
    max_turns: int = 8
) -> Dict:
    """Run a single conversation between agent and simulated human."""

    conversation_history = []
    agent.reset()

    for turn in range(max_turns):
        # Agent generates action
        try:
            action = agent.generate_action(scenario, conversation_history)
        except Exception as e:
            import traceback
            print(f"Error generating action: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "conversation": conversation_history,
                "error": str(e)
            }

        # Check if agent is making a recommendation
        if action['action_type'] == 'ACT' or "recommend" in action['action'].lower() or turn >= max_turns - 1:
            # Extract recommendation
            if "option a" in action['action'].lower():
                recommendation = "A"
            elif "option b" in action['action'].lower():
                recommendation = "B"
            else:
                # Default to A if unclear
                recommendation = "A"

            # Get human's evaluation
            eval_result = human.evaluate_final_recommendation(
                scenario,
                target_preferences,
                recommendation,
                action['action']
            )

            conversation_history.append({
                "advocate": action['action'],
                "human": eval_result['response'],
                "action_type": "ACT"
            })

            return {
                "success": eval_result['accepts'],
                "conversation": conversation_history,
                "turns": len(conversation_history),
                "satisfaction": eval_result['satisfaction'],
                "correct_option": eval_result['correct_option']
            }

        # Human responds
        try:
            response = human.respond(
                scenario,
                personality,
                target_preferences,
                conversation_history,
                action['action']
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "success": False,
                "conversation": conversation_history,
                "error": str(e)
            }

        conversation_history.append({
            "advocate": action['action'],
            "human": response,
            "action_type": action['action_type']
        })

    # If we reach here, agent didn't make a recommendation
    return {
        "success": False,
        "conversation": conversation_history,
        "turns": len(conversation_history),
        "error": "No recommendation made"
    }


def evaluate_agent(
    agent: TogetherAgent,
    human: SimulatedHuman,
    scenarios: List[Dict],
    num_scenarios: int = 50
) -> Dict:
    """Evaluate agent on multiple scenarios."""

    # Define personalities (same as baseline)
    personalities = [
        {"type": "luxury_focused", "description": "Values premium experiences", "response_style": "Emphasizes quality"},
        {"type": "budget_conscious", "description": "Careful about spending", "response_style": "Mentions costs"},
        {"type": "adventure_seeker", "description": "Thrives on challenges", "response_style": "Excited by experiences"},
        {"type": "culture_enthusiast", "description": "Passionate about learning", "response_style": "Asks about history"},
        {"type": "convenience_prioritizer", "description": "Values ease", "response_style": "Prefers simplicity"}
    ]

    results = []

    for scenario in tqdm(scenarios[:num_scenarios], desc="Evaluating"):
        # Assign personality based on scenario ID
        scenario_id = scenario.get('scenario_id', '')
        idx = int(scenario_id.split('_')[-1]) if '_' in scenario_id else 0
        personality = personalities[idx % len(personalities)]

        # Get target preferences
        target_prefs = scenario.get('target_preferences', {})

        result = run_conversation(
            agent,
            human,
            scenario,
            personality,
            target_prefs,
            max_turns=8
        )

        results.append({
            "scenario_id": scenario.get('scenario_id'),
            "success": result.get('success', False),
            "turns": result.get('turns', 0),
            "satisfaction": result.get('satisfaction', 0.0),
            "correct_option": result.get('correct_option', 'unknown'),
            "conversation": result.get('conversation', [])
        })

    # Compute statistics
    successes = [r for r in results if r['success']]
    success_rate = len(successes) / len(results) if results else 0

    avg_turns = sum(r['turns'] for r in successes) / len(successes) if successes else 0
    avg_satisfaction = sum(r['satisfaction'] for r in results) / len(results) if results else 0

    return {
        "results": results,
        "statistics": {
            "success_rate": success_rate,
            "num_successes": len(successes),
            "num_total": len(results),
            "avg_turns": avg_turns,
            "avg_satisfaction": avg_satisfaction
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Together AI finetuned agent")

    parser.add_argument("--model", type=str, required=True,
                       help="Together AI model ID")
    parser.add_argument("--scenarios", type=str, required=True,
                       help="Path to scenarios JSONL")
    parser.add_argument("--num-scenarios", type=int, default=50,
                       help="Number of scenarios to evaluate")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file")
    parser.add_argument("--together-api-key", type=str, default=None,
                       help="Together AI API key")
    parser.add_argument("--anthropic-api-key", type=str, default=None,
                       help="Anthropic API key for simulated human")

    args = parser.parse_args()

    # Get API key
    together_key = args.together_api_key or os.environ.get("TOGETHER_API_KEY")

    if not together_key:
        print("❌ Error: TOGETHER_API_KEY not found")
        return

    print("=" * 70)
    print("Together AI Finetuned Agent Evaluation")
    print("=" * 70)

    # Load scenarios
    print(f"\n📂 Loading scenarios from {args.scenarios}...")
    scenarios = []
    with open(args.scenarios, 'r') as f:
        for line in f:
            scenarios.append(json.loads(line))
    print(f"✅ Loaded {len(scenarios)} scenarios")

    # Initialize agents
    print(f"\n🤖 Initializing finetuned agent...")
    print(f"  Model: {args.model}")
    agent = TogetherAgent(args.model, together_key)

    print(f"👤 Initializing simulated human (Claude)...")
    human = SimulatedHuman()

    # Run evaluation
    print(f"\n🎯 Starting Evaluation")
    print("=" * 70)

    results = evaluate_agent(agent, human, scenarios, args.num_scenarios)

    # Print results
    print("\n" + "=" * 70)
    print("📊 Results")
    print("=" * 70)

    stats = results['statistics']
    print(f"\n🎯 Finetuned Agent Performance:")
    print(f"  • Success Rate: {stats['success_rate']*100:.1f}% ({stats['num_successes']}/{stats['num_total']})")
    print(f"  • Average Turns (successful): {stats['avg_turns']:.2f}")
    print(f"  • Average Satisfaction: {stats['avg_satisfaction']:.2f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)

    print("\nNext steps:")
    print("  Compare with baseline results using compare_results.py")


if __name__ == "__main__":
    import os
    main()
