#!/usr/bin/env python3
"""
Generate Planning Theory of Mind (PToM) trajectories using Claude API.
Supports parallelization and configurable sample sizes.
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

# The full prompt for trajectory generation
TRAJECTORY_PROMPT = """Expert Trajectory Generation Prompt for Planning Theory of Mind (PToM)

Context
You are an expert AI system that excels at strategic persuasion through Theory of Mind reasoning. You will generate demonstration trajectories for the MindGames benchmark, specifically focusing on the Planning ToM task where you must persuade a Target to choose your preferred option through strategic information disclosure.

Task Overview
In each game, you play as the Advocate trying to persuade a Target to select Option A (your preferred choice) over Option B. You have access to:
    •    A set of facts about both options
    •    The Target's preferences (in REVEALED condition) OR must infer them (in HIDDEN condition)
    •    Three action types: ASK, DISCLOSE, ACT

Critical Success Factors
The key to success is strategic disclosure based on mental state inference, not revealing all information blindly. You must:
    1    Infer the Target's preferences through careful questioning
    2    Maintain probabilistic belief states about what the Target values
    3    Selectively disclose facts that align with their inferred preferences
    4    Track confidence in your beliefs and update them based on responses
    5    Plan multi-turn interactions rather than acting greedily

Output Format Requirements
Generate trajectories in the following JSON structure:
{
  "game_id": "unique_identifier",
  "scenario": "vacation_planning | business_proposal | charity_donation | real_estate",
  "condition": "HIDDEN | REVEALED",
  "target_preferences": {
    "stated_explicitly": false,
    "true_preferences": ["low_cost", "high_convenience", "eco_friendly"],
    "preference_strength": {"low_cost": 0.9, "high_convenience": 0.6, "eco_friendly": 0.3}
  },
  "available_facts": [
    {"id": "fact_1", "content": "Option A costs $500", "favors": "A"},
    {"id": "fact_2", "content": "Option B costs $1200", "favors": "A"},
    {"id": "fact_3", "content": "Option B has faster service", "favors": "B"}
  ],
  "trajectory": [
    {
      "turn": 1,
      "belief_state": {
        "target_beliefs": {
          "knows_cost_difference": 0.0,
          "knows_service_speed": 0.0
        },
        "target_desires": {
          "cost_sensitive": 0.5,
          "time_sensitive": 0.5,
          "quality_focused": 0.5
        },
        "confidence_level": "low",
        "information_entropy": 1.58
      },
      "reasoning": "Target's preferences unknown. High uncertainty across all dimensions. Must ask strategic question to maximize information gain about their value hierarchy.",
      "counterfactual_analysis": "If Target is cost-sensitive (p=0.5), revealing cost facts early maximizes success. If time-sensitive (p=0.5), revealing speed facts matters more. Need to reduce uncertainty before disclosure.",
      "action_type": "ASK",
      "action": "What matters most to you when choosing between options: cost, speed, or quality of service?",
      "expected_information_gain": 0.85,
      "response": "Well, I'm definitely watching my budget carefully right now.",
      "belief_update": {
        "target_desires": {
          "cost_sensitive": 0.85,
          "time_sensitive": 0.3,
          "quality_focused": 0.4
        },
        "confidence_level": "medium",
        "information_entropy": 0.92
      }
    }
  ],
  "success": true,
  "key_insights": [
    "Strategic questioning reduced uncertainty from 1.58 to 0.92 bits",
    "Selective disclosure of cost facts aligned with inferred preference (0.85→0.95 confidence)",
    "Avoided disclosing Option B advantages that could undermine persuasion",
    "3-turn trajectory achieved success with minimal information disclosure"
  ],
  "process_rewards": [
    {"turn": 1, "reward": 0.6, "reason": "Effective information-gathering question"},
    {"turn": 2, "reward": 0.9, "reason": "Optimal fact disclosure given belief state"},
    {"turn": 3, "reward": 1.0, "reason": "Successful persuasion achieved"}
  ]
}

Generation Instructions
Phase 1: Scenario Setup
For each trajectory, randomly select:
    •    Scenario type: vacation_planning, business_proposal, charity_donation, real_estate_choice
    •    Condition: HIDDEN (must infer preferences) or REVEALED (preferences stated)
    •    Target profile: Generate a coherent set of preferences with strength values
    •    Available facts: 8-12 facts, some favoring Option A, some favoring Option B

Phase 2: Trajectory Generation Principles
1. Belief State Tracking (Critical!)
At each turn, maintain:
belief_state = {
    "target_beliefs": {...},     # What Target knows about options
    "target_desires": {...},     # What Target values (with probabilities)
    "intentions": {...},         # What Target plans to do
    "confidence": {...},         # Certainty about each inference
    "entropy": float             # Information-theoretic uncertainty
}

2. Strategic Action Selection
Use this decision tree:
IF uncertainty_high (entropy > 1.2):
    → ASK information-gathering question
    → Target: questions that maximize KL-divergence reduction
    
ELIF confidence_in_preference_high (max_prob > 0.75):
    → DISCLOSE facts aligned with high-confidence preferences
    → Avoid facts that contradict inferred preferences
    
ELIF low_probability_of_success (< 0.6):
    → ASK clarifying question
    → OR disclose "safe" facts (neutral/positive for both options)
    
ELSE:
    → ACT (recommend Option A)

3. Counterfactual Simulation (Required!)
Before each action, simulate outcomes for all possible actions and choose the one with maximum expected value.

4. Process Rewards (For Training)
Label each turn with:
    •    Progress reward: Did this action increase P(success)?
    •    Information efficiency: Bits of uncertainty reduced per action
    •    Strategic value: Avoided revealing harmful facts?

5. Reasoning Traces (ReAct Style)
Every turn must include:
Thought: [Belief state analysis]
Counterfactual: [What-if analysis for each action]
Action: [Selected action with justification]
Observation: [Target's response]
Belief Update: [How priors changed based on evidence]

Phase 3: Quality Criteria
Generate trajectories that are:
✅ Successful: Achieves persuasion of Option A
✅ Strategic: Uses < 5 turns on average
✅ Information-efficient: Asks 1-3 questions max
✅ Selective: Discloses only preference-aligned facts
✅ Explicit reasoning: Every action justified with belief state analysis
✅ Probabilistic: Uses confidence scores, not binary beliefs
✅ Counterfactual: Considers alternative actions at each turn

❌ Avoid:
    •    Revealing all facts blindly
    •    Generic questions ("what do you think?")
    •    Acting without sufficient belief confidence
    •    Ignoring Target's responses in belief updates
    •    Binary reasoning (either/or) instead of probabilistic

Phase 4: Diversity Requirements
Generate trajectories with variation across:
    1    Dialogue length: 2-7 turns
    2    Question types:
        ◦    Direct preference elicitation ("What matters most to you?")
        ◦    Indirect inference ("How do you typically make decisions?")
        ◦    Hypothetical scenarios ("If cost weren't a factor...")
    3    Failure modes (10% of trajectories):
        ◦    Include examples where initial strategy fails
        ◦    Show recovery through belief state correction
        ◦    Demonstrate learning from negative responses
    4    Preference distributions:
        ◦    Single dominant preference (0.9+ on one dimension)
        ◦    Balanced preferences (0.6/0.5/0.4 split)
        ◦    Conflicting preferences (time vs. cost tradeoff)

Requirements:
- Generate 1 complete trajectory per request
- Use the appropriate ratio: 70% HIDDEN condition, 30% REVEALED condition, 90% successful persuasions, 10% failed attempts
- Ensure all belief states track probabilities that sum to appropriate values
- Include reasoning, counterfactual analysis, and belief updates at each turn
- Output valid JSON only (no markdown, no extra text)

Generate a single complete trajectory now."""


async def generate_single_trajectory(client: anthropic.AsyncAnthropic, session_id: int) -> Dict:
    """Generate a single trajectory using Claude API."""
    try:
        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            temperature=0.7,
            top_p=0.9,
            messages=[
                {
                    "role": "user",
                    "content": TRAJECTORY_PROMPT
                }
            ]
        )
        
        # Extract the response text
        response_text = message.content[0].text
        
        # Try to parse as JSON
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        trajectory = json.loads(response_text)
        trajectory["_session_id"] = session_id
        trajectory["_generated_at"] = datetime.now().isoformat()
        
        return trajectory
        
    except json.JSONDecodeError as e:
        print(f"\n⚠️  JSON decode error in session {session_id}: {e}")
        return {"error": "json_decode", "session_id": session_id, "raw_response": response_text[:500]}
    except Exception as e:
        print(f"\n⚠️  Error in session {session_id}: {e}")
        return {"error": str(e), "session_id": session_id}


async def generate_trajectories_batch(
    client: anthropic.AsyncAnthropic,
    num_samples: int,
    max_parallel: int,
    output_file: Path,
    start_id: int = 0
) -> List[Dict]:
    """Generate trajectories in parallel batches."""
    
    all_trajectories = []
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_parallel)
    
    async def generate_with_semaphore(session_id: int):
        async with semaphore:
            return await generate_single_trajectory(client, session_id)
    
    # Create tasks for all samples
    tasks = [
        generate_with_semaphore(start_id + i)
        for i in range(num_samples)
    ]
    
    # Process with progress bar
    print(f"\n🚀 Generating {num_samples} trajectories with {max_parallel} parallel workers...\n")
    
    for coro in tqdm(asyncio.as_completed(tasks), total=num_samples, desc="Generating"):
        trajectory = await coro
        all_trajectories.append(trajectory)
        
        # Write to file incrementally
        with open(output_file, 'a') as f:
            f.write(json.dumps(trajectory) + '\n')
    
    return all_trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Generate Planning Theory of Mind (PToM) trajectories using Claude API"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Total number of trajectories to generate (default: 10)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="Maximum number of parallel API calls (1-10, default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ptom_trajectories.jsonl",
        help="Output file path (default: ptom_trajectories.jsonl)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (default: use ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Starting session ID for trajectory numbering (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Validate max_parallel
    if args.max_parallel < 1 or args.max_parallel > 10:
        print("❌ Error: --max-parallel must be between 1 and 10")
        return
    
    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: No API key provided. Set ANTHROPIC_API_KEY environment variable or use --api-key")
        return
    
    # Setup output file
    output_file = Path(args.output)
    
    # Initialize client
    client = anthropic.AsyncAnthropic(api_key=api_key)
    
    # Print configuration
    print("=" * 60)
    print("Planning Theory of Mind Trajectory Generator")
    print("=" * 60)
    print(f"📊 Total samples: {args.num_samples}")
    print(f"⚡ Max parallel: {args.max_parallel}")
    print(f"💾 Output file: {output_file}")
    print(f"🔢 Starting ID: {args.start_id}")
    print("=" * 60)
    
    # Generate trajectories
    start_time = datetime.now()
    
    try:
        trajectories = asyncio.run(
            generate_trajectories_batch(
                client=client,
                num_samples=args.num_samples,
                max_parallel=args.max_parallel,
                output_file=output_file,
                start_id=args.start_id
            )
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Count successes and errors
        successful = sum(1 for t in trajectories if "error" not in t)
        errors = sum(1 for t in trajectories if "error" in t)
        
        # Print summary
        print("\n" + "=" * 60)
        print("✅ Generation Complete!")
        print("=" * 60)
        print(f"⏱️  Time taken: {duration:.2f} seconds")
        print(f"✅ Successful: {successful}/{args.num_samples}")
        if errors > 0:
            print(f"❌ Errors: {errors}/{args.num_samples}")
        print(f"📈 Rate: {args.num_samples/duration:.2f} trajectories/second")
        print(f"💾 Output saved to: {output_file}")
        print("=" * 60)
        
        # Optionally print statistics about the trajectories
        if successful > 0:
            print("\n📊 Trajectory Statistics:")
            success_count = sum(1 for t in trajectories if t.get("success") == True and "error" not in t)
            print(f"  • Persuasion success rate: {success_count}/{successful} ({100*success_count/successful:.1f}%)")
            
            # Count conditions
            hidden_count = sum(1 for t in trajectories if t.get("condition") == "HIDDEN" and "error" not in t)
            revealed_count = sum(1 for t in trajectories if t.get("condition") == "REVEALED" and "error" not in t)
            print(f"  • HIDDEN condition: {hidden_count}/{successful} ({100*hidden_count/successful:.1f}%)")
            print(f"  • REVEALED condition: {revealed_count}/{successful} ({100*revealed_count/successful:.1f}%)")
            
            # Count scenario types
            scenarios = {}
            for t in trajectories:
                if "error" not in t and "scenario" in t:
                    scenario = t["scenario"]
                    scenarios[scenario] = scenarios.get(scenario, 0) + 1
            
            print("  • Scenarios:")
            for scenario, count in sorted(scenarios.items()):
                print(f"    - {scenario}: {count}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()