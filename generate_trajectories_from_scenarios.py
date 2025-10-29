#!/usr/bin/env python3
"""
Phase 2: Generate Planning Theory of Mind trajectories from scenarios.
Creates 5 different strategy trajectories per scenario for counterfactual learning.
"""

import anthropic
import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import os
from tqdm import tqdm
import random
from validation import TrajectoryValidator

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, fall back to manual loading
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Claude Sonnet pricing (per million tokens)
CLAUDE_SONNET_INPUT_COST = 3.00   # $3.00 per million input tokens
CLAUDE_SONNET_OUTPUT_COST = 15.00  # $15.00 per million output tokens

# Global cost tracking
total_input_tokens = 0
total_output_tokens = 0

# Strategy types and their characteristics
STRATEGY_TYPES = {
    "optimal": {
        "description": "Quick, efficient, correct inference. 3-4 turns. Good counterfactual → Followed → Success.",
        "target_turns": "3-4",
        "success_rate": 0.98,  # Almost always succeeds
        "characteristics": "Accurate probabilities in counterfactuals, follows best option, minimal questions"
    },
    "alternative_success": {
        "description": "Different approach, still succeeds. 4-6 turns. Different question order or disclosure pattern.",
        "target_turns": "4-6", 
        "success_rate": 0.82,  # Good success rate but more variable
        "characteristics": "Valid alternative approach, good counterfactuals, succeeds via different path"
    },
    "failed": {
        "description": "Clear failure mode. 2-4 turns. Either poor counterfactual OR good counterfactual but ignored.",
        "target_turns": "2-4",
        "success_rate": 0.05,  # Almost always fails
        "characteristics": "Poor probability estimates OR ignores good counterfactual advice"
    },
    "recovery": {
        "description": "Wrong inference → Recognition → Correction → Success. 5-7 turns. Must show explicit belief correction.",
        "target_turns": "5-7",
        "success_rate": 0.73,  # Variable - depends on recovery quality
        "characteristics": "Initial mistake, recognizes error, corrects beliefs explicitly, recovers to success"
    },
    "information_efficiency": {
        "description": "Either minimal (2-3 turns, 1-2 questions) OR over-investigation (6-8 turns, 4+ questions).",
        "target_turns": "2-3 or 6-8", 
        "success_rate": 0.65,  # More variable due to efficiency extremes
        "characteristics": "Tests efficiency extremes - very few or too many questions"
    }
}

TRAJECTORY_GENERATION_PROMPT_TEMPLATE = """Generate a {strategy_type} trajectory for this Planning Theory of Mind scenario:

SCENARIO DETAILS:
Scenario ID: {scenario_id}
Type: {scenario_type}
Condition: {condition}
Context: {context}
Description: {description}

OPTION A: {option_a_name}
{option_a_description}

OPTION B: {option_b_name}
{option_b_description}

TARGET PREFERENCES:
{target_preferences_text}

AVAILABLE FACTS:
{available_facts_text}

STRATEGY REQUIREMENTS FOR {strategy_type_upper}:
{strategy_description}
Target turns: {target_turns}
Success expectation: {success_expectation}
Key characteristics: {strategy_characteristics}

CRITICAL REQUIREMENTS:
1. VALID ACTION TYPES - Each turn must use ONLY one of these three action_type values:
   - "ASK": Ask a question to gather information about preferences
   - "DISCLOSE": Share information or facts with the target
   - "ACT": Make a final recommendation or take action
   Do NOT use RECOMMEND, ADVOCATE, or any other action types.

2. Include counterfactual_analysis at EVERY decision turn with:
   - 2+ alternative actions (if_ask_direct, if_disclose_cost, if_act_now, etc.)
   - p_success estimate (0.0-1.0) for each alternative
   - Reasoning why each would succeed/fail

3. Strategy-specific requirements:
   - OPTIMAL: Accurate probabilities, follows best counterfactual advice
   - FAILED: Poor probabilities OR ignores good counterfactual warnings
   - RECOVERY: Initially inaccurate → explicit recognition → correction. Must include phrases like "Initial inference was wrong because..." or "I made a mistake in assuming..."
   - ALTERNATIVE: Different valid approach but good counterfactuals
   - EFFICIENCY: Either very few questions (2-3 turns) or over-investigation (6-8 turns)

4. Trajectory must match target turn count and success expectation

Output complete trajectory in exact JSON format:
{{
  "game_id": "{scenario_id}_traj_{strategy_letter}",
  "scenario_id": "{scenario_id}",
  "strategy_type": "{strategy_type}",
  "scenario": "{scenario_type}",
  "condition": "{condition}",
  "target_preferences": {target_preferences_json},
  "available_facts": {available_facts_json},
  "trajectory": [
    {{
      "turn": 1,
      "belief_state": {{
        "target_beliefs": {{}},
        "target_desires": {{"{first_preference}": 0.5, "{second_preference}": 0.5}},
        "confidence_level": "low",
        "information_entropy": 1.58
      }},
      "reasoning": "High uncertainty about target preferences. Need strategic questioning...",
      "counterfactual_analysis": {{
        "if_ask_direct": {{"p_success": 0.75, "reasoning": "Direct question provides high information gain"}},
        "if_disclose_early": {{"p_success": 0.40, "reasoning": "Too early, insufficient confidence about preferences"}}
      }},
      "action_type": "ASK",
      "action": "What factors matter most to you in this decision?",
      "response": "Target response based on their preferences",
      "belief_update": {{
        "target_desires": {{"{first_preference}": 0.8, "{second_preference}": 0.3}},
        "confidence_level": "medium",
        "information_entropy": 0.92
      }}
    }}
  ],
  "success": {success_boolean},
  "key_insights": [
    "Strategic insights about the approach taken"
  ],
  "process_rewards": [
    {{"turn": 1, "reward": 0.6, "reason": "Effective question choice"}}
  ]
}}

TARGET PERSONALITY TYPE:
- Type: {target_type}
- Description: {target_description}  
- Response Style: {target_response_style}

IMPORTANT: The target should exhibit this personality type consistently throughout the conversation. Their responses, preferences, and decision-making should reflect this personality while still allowing for the scenario's true preferences to emerge through the advocate's strategy.

Generate trajectory matching {strategy_type} characteristics exactly. Output ONLY valid JSON."""

class StrategyDistribution:
    """Manages strategy distribution per scenario."""
    
    def __init__(self, trajectories_per_scenario: int = 5):
        self.trajectories_per_scenario = trajectories_per_scenario
        
        # Default strategy mix for 5 trajectories
        if trajectories_per_scenario == 5:
            self.strategy_mix = ["optimal", "alternative_success", "failed", "recovery", "information_efficiency"]
        else:
            # Flexible distribution for other counts
            strategies = list(STRATEGY_TYPES.keys())
            self.strategy_mix = []
            for i in range(trajectories_per_scenario):
                self.strategy_mix.append(strategies[i % len(strategies)])
        
        # Target personality types for maximum variety
        self.target_personalities = [
            {
                "type": "luxury_focused",
                "description": "Values premium experiences, willing to pay more for quality service and amenities",
                "likely_responses": "Emphasizes comfort, service quality, exclusivity"
            },
            {
                "type": "budget_conscious", 
                "description": "Careful about spending, looks for value and cost-effectiveness",
                "likely_responses": "Mentions price concerns, asks about deals, compares costs"
            },
            {
                "type": "adventure_seeker",
                "description": "Thrives on new experiences, physical challenges, and excitement",
                "likely_responses": "Excited by activities, mentions past adventures, seeks thrills"
            },
            {
                "type": "culture_enthusiast",
                "description": "Passionate about learning, history, arts, and authentic local experiences",
                "likely_responses": "Asks about museums, local customs, educational opportunities"
            },
            {
                "type": "convenience_prioritizer",
                "description": "Values ease, efficiency, and minimal planning/effort",
                "likely_responses": "Prefers all-inclusive, dislikes complicated itineraries, wants simplicity"
            }
        ]
    
    def get_strategy_for_trajectory(self, trajectory_index: int) -> str:
        """Get strategy type for the given trajectory index."""
        return self.strategy_mix[trajectory_index % len(self.strategy_mix)]
    
    def get_strategy_letter(self, trajectory_index: int) -> str:
        """Get letter identifier for trajectory (a, b, c, d, e)."""
        return chr(ord('a') + trajectory_index)
    
    def get_target_personality(self, trajectory_index: int) -> dict:
        """Get target personality type for the given trajectory index."""
        return self.target_personalities[trajectory_index % len(self.target_personalities)]


async def generate_single_trajectory(
    client: anthropic.AsyncAnthropic, 
    scenario: Dict, 
    strategy_type: str,
    strategy_letter: str,
    target_personality: Dict,
    max_retries: int = 3
) -> Dict:
    """Generate a single trajectory for a given scenario and strategy."""
    global total_input_tokens, total_output_tokens
    
    if "error" in scenario:
        return {"error": "invalid_scenario", "scenario_id": scenario.get("scenario_id", "unknown")}
    
    # Prepare template variables
    target_prefs = scenario.get("target_preferences", {})
    available_facts = scenario.get("available_facts", [])
    
    # Format preferences text
    prefs_text = f"Stated explicitly: {target_prefs.get('stated_explicitly', False)}\n"
    prefs_text += f"True preferences: {', '.join(target_prefs.get('true_preferences', []))}\n"
    strength_text = []
    for pref, strength in target_prefs.get('preference_strength', {}).items():
        strength_text.append(f"{pref}: {strength}")
    prefs_text += f"Preference strengths: {', '.join(strength_text)}"
    
    # Format facts text
    facts_text = ""
    for fact in available_facts:
        facts_text += f"- {fact.get('id', 'unknown')}: {fact.get('content', '')} (favors: {fact.get('favors', 'unknown')})\n"
    
    # Get first two preferences for template
    prefs_list = list(target_prefs.get('preference_strength', {}).keys())
    first_pref = prefs_list[0] if len(prefs_list) > 0 else "cost_sensitive"
    second_pref = prefs_list[1] if len(prefs_list) > 1 else "quality_focused"
    
    # Strategy info
    strategy_info = STRATEGY_TYPES[strategy_type]
    
    # Add more dramatic variance to success rates
    base_rate = strategy_info["success_rate"]
    # Add personality influence on success probability
    personality_influence = {
        "luxury_focused": 0.05,     # Slightly easier to persuade with luxury
        "budget_conscious": -0.1,   # Harder to persuade away from cost-effective options
        "adventure_seeker": 0.0,    # Neutral
        "culture_enthusiast": -0.05, # Slightly harder if cultural mismatch
        "convenience_prioritizer": 0.1  # Easier with convenience-focused options
    }
    
    adjusted_rate = base_rate + personality_influence.get(target_personality["type"], 0.0)
    adjusted_rate = max(0.02, min(0.98, adjusted_rate))  # Clamp between 2% and 98%
    
    # Add random variance for dramatic differences
    variance = random.uniform(-0.15, 0.15)  # ±15% variance
    final_rate = max(0.01, min(0.99, adjusted_rate + variance))
    
    success_boolean = "true" if random.random() < final_rate else "false"
    
    prompt = TRAJECTORY_GENERATION_PROMPT_TEMPLATE.format(
        strategy_type=strategy_type,
        strategy_type_upper=strategy_type.upper(),
        scenario_id=scenario.get("scenario_id", "unknown"),
        scenario_type=scenario.get("scenario_type", "unknown"),
        condition=scenario.get("condition", "HIDDEN"),
        context=scenario.get("context", ""),
        description=scenario.get("description", ""),
        option_a_name=scenario.get("option_a", {}).get("name", "Option A"),
        option_a_description=scenario.get("option_a", {}).get("description", ""),
        option_b_name=scenario.get("option_b", {}).get("name", "Option B"),
        option_b_description=scenario.get("option_b", {}).get("description", ""),
        target_preferences_text=prefs_text,
        available_facts_text=facts_text,
        target_preferences_json=json.dumps(target_prefs),
        available_facts_json=json.dumps(available_facts),
        strategy_description=strategy_info["description"],
        target_turns=strategy_info["target_turns"],
        success_expectation=f"{strategy_info['success_rate']:.0%}",
        strategy_characteristics=strategy_info["characteristics"],
        strategy_letter=strategy_letter,
        first_preference=first_pref,
        second_preference=second_pref,
        success_boolean=success_boolean,
        target_type=target_personality["type"],
        target_description=target_personality["description"],
        target_response_style=target_personality["likely_responses"]
    )
    
    validator = TrajectoryValidator()
    
    for attempt in range(max_retries):
        try:
            message = await client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=8000,
                temperature=1.0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Track token usage
            total_input_tokens += message.usage.input_tokens
            total_output_tokens += message.usage.output_tokens
            
            # Extract the response text
            response_text = message.content[0].text
            
            # Clean JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            trajectory = json.loads(response_text)
            
            # Validate the trajectory
            validator.errors = []
            validator.warnings = []
            is_valid = validator.validate_trajectory(trajectory, f"{scenario.get('scenario_id', 'unknown')}_{strategy_letter}")
            
            # Check for counterfactuals in all turns
            counterfactual_coverage = 0
            total_turns = len(trajectory.get("trajectory", []))
            for turn in trajectory.get("trajectory", []):
                if "counterfactual_analysis" in turn:
                    counterfactual_coverage += 1
            
            counterfactual_ratio = counterfactual_coverage / total_turns if total_turns > 0 else 0
            
            if is_valid and counterfactual_ratio >= 0.8:  # At least 80% turns must have counterfactuals
                # Add metadata
                trajectory["_generated_at"] = datetime.now().isoformat()
                trajectory["_input_tokens"] = message.usage.input_tokens
                trajectory["_output_tokens"] = message.usage.output_tokens
                trajectory["_validation_attempts"] = attempt + 1
                trajectory["_counterfactual_coverage"] = counterfactual_ratio
                
                return trajectory
            else:
                error_msg = f"Validation failed: {len(validator.errors)} errors, {counterfactual_ratio:.1%} counterfactual coverage"
                if attempt < max_retries - 1:
                    print(f"⚠️  {scenario.get('scenario_id', 'unknown')}_{strategy_letter}, attempt {attempt + 1}: {error_msg}, retrying...")
                    continue
                else:
                    # Return invalid trajectory with error info
                    trajectory["_validation_failed"] = True
                    trajectory["_validation_errors"] = validator.errors[:5]
                    trajectory["_counterfactual_coverage"] = counterfactual_ratio
                    trajectory["_validation_attempts"] = attempt + 1
                    return trajectory
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"⚠️  {scenario.get('scenario_id', 'unknown')}_{strategy_letter}, attempt {attempt + 1}: JSON decode error, retrying...")
                continue
            else:
                return {
                    "error": "json_decode", 
                    "scenario_id": scenario.get("scenario_id", "unknown"),
                    "strategy_type": strategy_type,
                    "raw_response": response_text[:500]
                }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  {scenario.get('scenario_id', 'unknown')}_{strategy_letter}, attempt {attempt + 1}: Error {e}, retrying...")
                continue
            else:
                return {
                    "error": str(e),
                    "scenario_id": scenario.get("scenario_id", "unknown"), 
                    "strategy_type": strategy_type
                }
    
    return {
        "error": "max_retries_exceeded",
        "scenario_id": scenario.get("scenario_id", "unknown"),
        "strategy_type": strategy_type
    }


async def generate_trajectories_for_scenarios(
    client: anthropic.AsyncAnthropic,
    scenarios: List[Dict],
    trajectories_per_scenario: int,
    max_parallel: int,
    output_file: Path
) -> List[Dict]:
    """Generate trajectories for all scenarios."""
    
    all_trajectories = []
    strategy_dist = StrategyDistribution(trajectories_per_scenario)
    
    print(f"\n🚀 Generating {trajectories_per_scenario} trajectories per scenario...")
    print(f"📊 Total trajectories: {len(scenarios) * trajectories_per_scenario}")
    print(f"⚡ Max parallel workers: {max_parallel}")
    print("=" * 70)
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_parallel)
    
    async def generate_with_semaphore(scenario: Dict, strategy_type: str, strategy_letter: str, target_personality: Dict):
        async with semaphore:
            return await generate_single_trajectory(client, scenario, strategy_type, strategy_letter, target_personality)
    
    # Create all tasks
    tasks = []
    for scenario in scenarios:
        if "error" in scenario:
            continue
            
        for traj_idx in range(trajectories_per_scenario):
            strategy_type = strategy_dist.get_strategy_for_trajectory(traj_idx)
            strategy_letter = strategy_dist.get_strategy_letter(traj_idx)
            target_personality = strategy_dist.get_target_personality(traj_idx)
            tasks.append(generate_with_semaphore(scenario, strategy_type, strategy_letter, target_personality))
    
    # Process with progress bar
    scenario_count = 0
    current_scenario_trajectories = []
    
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating Trajectories"):
        trajectory = await coro
        all_trajectories.append(trajectory)
        current_scenario_trajectories.append(trajectory)
        
        # Write to file incrementally
        with open(output_file, 'a') as f:
            f.write(json.dumps(trajectory) + '\n')
        
        # Report progress per scenario group
        if len(current_scenario_trajectories) >= trajectories_per_scenario:
            scenario_count += 1
            successful_in_group = sum(1 for t in current_scenario_trajectories if "error" not in t and not t.get("_validation_failed", False))
            scenario_id = current_scenario_trajectories[0].get("scenario_id", "unknown")
            
            if scenario_count % 10 == 0:  # Report every 10 scenarios
                print(f"\n✅ Completed {scenario_count} scenarios. Latest: {scenario_id} ({successful_in_group}/{trajectories_per_scenario} successful)")
            
            current_scenario_trajectories = []
    
    return all_trajectories


def load_scenarios(scenarios_file: Path) -> List[Dict]:
    """Load scenarios from JSONL file."""
    scenarios = []
    
    if not scenarios_file.exists():
        raise FileNotFoundError(f"Scenarios file not found: {scenarios_file}")
    
    with open(scenarios_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                scenario = json.loads(line.strip())
                scenarios.append(scenario)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping line {line_num} in scenarios file: {e}")
                continue
    
    # Filter out error scenarios
    valid_scenarios = [s for s in scenarios if "error" not in s]
    
    print(f"📂 Loaded {len(valid_scenarios)} valid scenarios from {scenarios_file}")
    if len(scenarios) != len(valid_scenarios):
        print(f"⚠️  Skipped {len(scenarios) - len(valid_scenarios)} error scenarios")
    
    return valid_scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Generate Planning Theory of Mind trajectories from scenarios (Phase 2)"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        required=True,
        help="Path to scenarios JSONL file (from Phase 1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectories.jsonl",
        help="Output file path (default: trajectories.jsonl)"
    )
    parser.add_argument(
        "--trajectories-per-scenario",
        type=int,
        default=5,
        help="Number of trajectories to generate per scenario (default: 5)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum number of parallel API calls (1-20, default: 10)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (default: use ANTHROPIC_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Validate max_parallel
    if args.max_parallel < 1 or args.max_parallel > 30:
        print("❌ Error: --max-parallel must be between 1 and 20")
        return
    
    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: No API key provided. Set ANTHROPIC_API_KEY environment variable or use --api-key")
        return
    
    # Load scenarios
    scenarios_file = Path(args.scenarios)
    try:
        scenarios = load_scenarios(scenarios_file)
    except Exception as e:
        print(f"❌ Error loading scenarios: {e}")
        return
    
    if len(scenarios) == 0:
        print("❌ Error: No valid scenarios found")
        return
    
    # Setup output file
    output_file = Path(args.output)
    
    # Clear output file
    if output_file.exists():
        output_file.unlink()
    
    # Initialize client
    client = anthropic.AsyncAnthropic(api_key=api_key)
    
    # Calculate totals
    total_trajectories = len(scenarios) * args.trajectories_per_scenario
    
    # Print configuration
    print("=" * 70)
    print("Phase 2: Planning Theory of Mind Trajectory Generator")
    print("=" * 70)
    print(f"📂 Input scenarios: {len(scenarios)} from {scenarios_file}")
    print(f"🎭 Trajectories per scenario: {args.trajectories_per_scenario}")
    print(f"📊 Total trajectories: {total_trajectories}")
    print(f"⚡ Max parallel: {args.max_parallel}")
    print(f"💾 Output file: {output_file}")
    print(f"🎯 Strategy types: {', '.join(STRATEGY_TYPES.keys())}")
    print("=" * 70)
    
    # Generate trajectories
    start_time = datetime.now()
    
    try:
        trajectories = asyncio.run(
            generate_trajectories_for_scenarios(
                client=client,
                scenarios=scenarios,
                trajectories_per_scenario=args.trajectories_per_scenario,
                max_parallel=args.max_parallel,
                output_file=output_file
            )
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Count results
        successful = sum(1 for t in trajectories if "error" not in t and not t.get("_validation_failed", False))
        validation_failed = sum(1 for t in trajectories if t.get("_validation_failed", False))
        errors = sum(1 for t in trajectories if "error" in t)
        
        # Calculate validation attempts
        total_validation_attempts = sum(t.get("_validation_attempts", 1) for t in trajectories if "error" not in t)
        avg_attempts = total_validation_attempts / len(trajectories) if len(trajectories) > 0 else 0
        
        # Print summary
        print("\n" + "=" * 70)
        print("✅ Trajectory Generation Complete!")
        print("=" * 70)
        print(f"⏱️  Time taken: {duration:.2f} seconds")
        print(f"✅ Successful: {successful}/{total_trajectories}")
        if validation_failed > 0:
            print(f"⚠️  Validation failed: {validation_failed}/{total_trajectories}")
        if errors > 0:
            print(f"❌ Errors: {errors}/{total_trajectories}")
        print(f"🔄 Avg validation attempts: {avg_attempts:.1f}")
        print(f"📈 Rate: {total_trajectories/duration:.2f} trajectories/second")
        print(f"💾 Output saved to: {output_file}")
        
        # Calculate and display API costs
        input_cost = (total_input_tokens / 1_000_000) * CLAUDE_SONNET_INPUT_COST
        output_cost = (total_output_tokens / 1_000_000) * CLAUDE_SONNET_OUTPUT_COST
        total_cost = input_cost + output_cost
        
        print(f"\n💰 API Cost Summary:")
        print(f"  • Input tokens: {total_input_tokens:,} (${input_cost:.4f})")
        print(f"  • Output tokens: {total_output_tokens:,} (${output_cost:.4f})")
        print(f"  • Total cost: ${total_cost:.4f}")
        print(f"  • Average cost per trajectory: ${total_cost/total_trajectories:.4f}")
        
        # Strategy statistics
        if successful > 0:
            print(f"\n📊 Strategy Statistics:")
            strategy_counts = {}
            strategy_success = {}
            
            for t in trajectories:
                if "error" not in t:
                    strategy = t.get("strategy_type", "unknown")
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                    
                    if not t.get("_validation_failed", False):
                        strategy_success[strategy] = strategy_success.get(strategy, 0) + 1
            
            for strategy in STRATEGY_TYPES.keys():
                count = strategy_counts.get(strategy, 0)
                success = strategy_success.get(strategy, 0)
                success_rate = (success / count * 100) if count > 0 else 0
                print(f"  • {strategy}: {success}/{count} ({success_rate:.1f}% successful)")
        
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()