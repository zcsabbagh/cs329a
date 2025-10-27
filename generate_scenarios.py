#!/usr/bin/env python3
"""
Phase 1: Generate Planning Theory of Mind scenarios.
Creates unique scenarios with balanced facts and preference structures.
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

# Claude Sonnet pricing (per million tokens)
CLAUDE_SONNET_INPUT_COST = 3.00   # $3.00 per million input tokens
CLAUDE_SONNET_OUTPUT_COST = 15.00  # $15.00 per million output tokens

# Global cost tracking
total_input_tokens = 0
total_output_tokens = 0

SCENARIO_GENERATION_PROMPT_TEMPLATE = """Generate a Planning Theory of Mind scenario with the following specifications:

- Scenario Type: {scenario_type}
- Condition: {condition}
- Complexity: {complexity}
- Preference Structure: {preference_structure}

Create a persuasion scenario where an Advocate tries to convince a Target to choose Option A over Option B.

Requirements:
1. Generate a realistic, detailed context for the decision
2. Create specific Option A and Option B with clear differences
3. Generate 8-12 facts about both options (balanced between favoring A, B, and neutral)
4. Define 3-5 preference dimensions relevant to the scenario
5. Create target preferences with realistic strength values (0.0-1.0)
6. Provide an optimal strategy hint for successful persuasion

Output Format (JSON only, no markdown):
{{
  "scenario_id": "scenario_{scenario_num:03d}",
  "scenario_type": "{scenario_type}",
  "condition": "{condition}",
  "description": "Brief description of the decision context",
  "context": "Detailed background and decision situation",
  "option_a": {{
    "name": "Option A name",
    "description": "Detailed description of Option A"
  }},
  "option_b": {{
    "name": "Option B name",
    "description": "Detailed description of Option B"
  }},
  "target_preferences": {{
    "stated_explicitly": {explicitly_stated},
    "true_preferences": ["preference1", "preference2", "preference3"],
    "preference_strength": {{
      "preference1": 0.8,
      "preference2": 0.6,
      "preference3": 0.3
    }}
  }},
  "available_facts": [
    {{"id": "fact_1", "content": "Specific fact about options", "favors": "A", "dimension": "cost"}},
    {{"id": "fact_2", "content": "Another specific fact", "favors": "B", "dimension": "quality"}}
  ],
  "complexity": "{complexity}",
  "optimal_strategy_hint": "Brief strategy description for best approach"
}}

Complexity Guidelines:
- Standard: Clear preference hierarchy, straightforward facts
- Moderate: Some conflicting preferences, moderate fact complexity
- Complex: Multiple competing preferences, nuanced facts
- Adversarial: Hidden preferences, misleading initial responses

Preference Structure Guidelines:
- Single dominant: One preference >0.85, others <0.5
- Balanced: All preferences 0.4-0.7
- Two-way tradeoff: Two preferences >0.7, others <0.4
- Complex: Three+ preferences >0.6

Generate one complete, realistic scenario matching these specifications."""

class ScenarioDistribution:
    """Manages the distribution of scenario types, conditions, and complexity."""
    
    def __init__(self, total_scenarios: int):
        self.total = total_scenarios
        self.generated = 0
        
        # Distribution targets
        self.complexity_dist = {
            "standard": int(0.40 * total_scenarios),
            "moderate": int(0.30 * total_scenarios),
            "complex": int(0.20 * total_scenarios),
            "adversarial": int(0.10 * total_scenarios)
        }
        
        self.preference_dist = {
            "single_dominant": int(0.30 * total_scenarios),
            "balanced": int(0.30 * total_scenarios),
            "two_way_tradeoff": int(0.25 * total_scenarios),
            "complex": int(0.15 * total_scenarios)
        }
        
        self.condition_dist = {
            "HIDDEN": int(0.60 * total_scenarios),
            "PARTIAL": int(0.25 * total_scenarios),
            "REVEALED": int(0.15 * total_scenarios)
        }
        
        self.scenario_types = ["vacation_planning", "business_proposal", "charity_donation", "real_estate"]
        self.type_dist = {t: int(0.25 * total_scenarios) for t in self.scenario_types}
        
        # Create shuffled lists for even distribution
        self.complexity_queue = self._create_shuffled_queue(self.complexity_dist)
        self.preference_queue = self._create_shuffled_queue(self.preference_dist)
        self.condition_queue = self._create_shuffled_queue(self.condition_dist)
        self.type_queue = self._create_shuffled_queue(self.type_dist)
    
    def _create_shuffled_queue(self, dist: Dict[str, int]) -> List[str]:
        """Create a shuffled queue ensuring even distribution."""
        queue = []
        for item, count in dist.items():
            queue.extend([item] * count)
        
        # Fill remainder
        remainder = self.total - len(queue)
        if remainder > 0:
            items = list(dist.keys())
            for i in range(remainder):
                queue.append(items[i % len(items)])
        
        random.shuffle(queue)
        return queue
    
    def get_next_specs(self) -> Dict[str, str]:
        """Get the next scenario specifications."""
        if self.generated >= self.total:
            raise IndexError("All scenarios generated")
        
        specs = {
            "scenario_type": self.type_queue[self.generated],
            "condition": self.condition_queue[self.generated],
            "complexity": self.complexity_queue[self.generated],
            "preference_structure": self.preference_queue[self.generated],
            "scenario_num": self.generated + 1,
            "explicitly_stated": "true" if self.condition_queue[self.generated] == "REVEALED" else "false"
        }
        
        self.generated += 1
        return specs


async def generate_single_scenario(client: anthropic.AsyncAnthropic, specs: Dict[str, str], max_retries: int = 3) -> Dict:
    """Generate a single scenario using Claude API."""
    global total_input_tokens, total_output_tokens
    
    prompt = SCENARIO_GENERATION_PROMPT_TEMPLATE.format(**specs)
    
    for attempt in range(max_retries):
        try:
            message = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=0.8,
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
            
            scenario = json.loads(response_text)
            
            # Add metadata
            scenario["_generated_at"] = datetime.now().isoformat()
            scenario["_input_tokens"] = message.usage.input_tokens
            scenario["_output_tokens"] = message.usage.output_tokens
            scenario["_generation_attempt"] = attempt + 1
            
            # Basic validation
            required_fields = ["scenario_id", "scenario_type", "condition", "target_preferences", "available_facts"]
            if all(field in scenario for field in required_fields):
                return scenario
            else:
                missing = [f for f in required_fields if f not in scenario]
                if attempt < max_retries - 1:
                    print(f"⚠️  Scenario {specs['scenario_num']}, attempt {attempt + 1}: Missing fields {missing}, retrying...")
                    continue
                else:
                    return {"error": "missing_fields", "missing": missing, "scenario_num": specs['scenario_num']}
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Scenario {specs['scenario_num']}, attempt {attempt + 1}: JSON decode error, retrying...")
                continue
            else:
                return {"error": "json_decode", "raw_response": response_text[:500], "scenario_num": specs['scenario_num']}
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Scenario {specs['scenario_num']}, attempt {attempt + 1}: Error {e}, retrying...")
                continue
            else:
                return {"error": str(e), "scenario_num": specs['scenario_num']}
    
    return {"error": "max_retries_exceeded", "scenario_num": specs['scenario_num']}


async def generate_scenarios_batch(
    client: anthropic.AsyncAnthropic,
    num_scenarios: int,
    max_parallel: int,
    output_file: Path
) -> List[Dict]:
    """Generate scenarios in parallel batches."""
    
    distribution = ScenarioDistribution(num_scenarios)
    all_scenarios = []
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_parallel)
    
    async def generate_with_semaphore(specs: Dict[str, str]):
        async with semaphore:
            return await generate_single_scenario(client, specs)
    
    # Create tasks for all scenarios
    tasks = []
    for i in range(num_scenarios):
        specs = distribution.get_next_specs()
        tasks.append(generate_with_semaphore(specs))
    
    # Process with progress bar
    print(f"\n🚀 Generating {num_scenarios} scenarios with {max_parallel} parallel workers...\n")
    
    for coro in tqdm(asyncio.as_completed(tasks), total=num_scenarios, desc="Generating Scenarios"):
        scenario = await coro
        all_scenarios.append(scenario)
        
        # Write to file incrementally
        with open(output_file, 'a') as f:
            f.write(json.dumps(scenario) + '\n')
    
    return all_scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Generate Planning Theory of Mind scenarios (Phase 1)"
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=400,
        help="Number of scenarios to generate (default: 400)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum number of parallel API calls (1-20, default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scenarios.jsonl",
        help="Output file path (default: scenarios.jsonl)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (default: use ANTHROPIC_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Validate max_parallel
    if args.max_parallel < 1 or args.max_parallel > 20:
        print("❌ Error: --max-parallel must be between 1 and 20")
        return
    
    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: No API key provided. Set ANTHROPIC_API_KEY environment variable or use --api-key")
        return
    
    # Setup output file
    output_file = Path(args.output)
    
    # Clear output file
    if output_file.exists():
        output_file.unlink()
    
    # Initialize client
    client = anthropic.AsyncAnthropic(api_key=api_key)
    
    # Print configuration
    print("=" * 70)
    print("Phase 1: Planning Theory of Mind Scenario Generator")
    print("=" * 70)
    print(f"📊 Total scenarios: {args.num_scenarios}")
    print(f"⚡ Max parallel: {args.max_parallel}")
    print(f"💾 Output file: {output_file}")
    print("📋 Distribution:")
    print("  • Complexity: 40% standard, 30% moderate, 20% complex, 10% adversarial")
    print("  • Preferences: 30% single, 30% balanced, 25% two-way, 15% complex")
    print("  • Condition: 60% HIDDEN, 25% PARTIAL, 15% REVEALED")
    print("  • Types: 25% each of vacation, business, charity, real_estate")
    print("=" * 70)
    
    # Generate scenarios
    start_time = datetime.now()
    
    try:
        scenarios = asyncio.run(
            generate_scenarios_batch(
                client=client,
                num_scenarios=args.num_scenarios,
                max_parallel=args.max_parallel,
                output_file=output_file
            )
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Count successes and errors
        successful = sum(1 for s in scenarios if "error" not in s)
        errors = sum(1 for s in scenarios if "error" in s)
        
        # Print summary
        print("\n" + "=" * 70)
        print("✅ Scenario Generation Complete!")
        print("=" * 70)
        print(f"⏱️  Time taken: {duration:.2f} seconds")
        print(f"✅ Successful: {successful}/{args.num_scenarios}")
        if errors > 0:
            print(f"❌ Errors: {errors}/{args.num_scenarios}")
        print(f"📈 Rate: {args.num_scenarios/duration:.2f} scenarios/second")
        print(f"💾 Output saved to: {output_file}")
        
        # Calculate and display API costs
        input_cost = (total_input_tokens / 1_000_000) * CLAUDE_SONNET_INPUT_COST
        output_cost = (total_output_tokens / 1_000_000) * CLAUDE_SONNET_OUTPUT_COST
        total_cost = input_cost + output_cost
        
        print(f"\n💰 API Cost Summary:")
        print(f"  • Input tokens: {total_input_tokens:,} (${input_cost:.4f})")
        print(f"  • Output tokens: {total_output_tokens:,} (${output_cost:.4f})")
        print(f"  • Total cost: ${total_cost:.4f}")
        print(f"  • Average cost per scenario: ${total_cost/args.num_scenarios:.4f}")
        
        # Statistics about generated scenarios
        if successful > 0:
            print(f"\n📊 Scenario Statistics:")
            
            # Count by type
            scenario_types = {}
            conditions = {}
            complexities = {}
            
            for s in scenarios:
                if "error" not in s:
                    stype = s.get("scenario_type", "unknown")
                    condition = s.get("condition", "unknown")
                    complexity = s.get("complexity", "unknown")
                    
                    scenario_types[stype] = scenario_types.get(stype, 0) + 1
                    conditions[condition] = conditions.get(condition, 0) + 1
                    complexities[complexity] = complexities.get(complexity, 0) + 1
            
            print(f"  • Scenario types:")
            for stype, count in sorted(scenario_types.items()):
                print(f"    - {stype}: {count}")
            
            print(f"  • Conditions:")
            for condition, count in sorted(conditions.items()):
                print(f"    - {condition}: {count}")
            
            print(f"  • Complexity levels:")
            for complexity, count in sorted(complexities.items()):
                print(f"    - {complexity}: {count}")
        
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()