#!/usr/bin/env python3
"""
Post-process trajectories by merging back scenario context.
Takes generated trajectories and original scenarios, outputs enriched trajectories.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def load_scenarios(scenario_file: Path) -> Dict[str, Dict]:
    """Load scenarios into a lookup dictionary by scenario_id."""
    scenarios = {}

    with open(scenario_file, 'r') as f:
        for line in f:
            scenario = json.loads(line.strip())
            scenario_id = scenario.get("scenario_id")
            if scenario_id:
                scenarios[scenario_id] = scenario

    return scenarios


def enrich_trajectory(trajectory: Dict, scenario: Dict) -> Dict:
    """Enrich trajectory with scenario context."""

    # Add scenario context fields
    trajectory["scenario_type"] = scenario.get("scenario_type", "unknown")
    trajectory["description"] = scenario.get("description", "")
    trajectory["context"] = scenario.get("context", "")

    # Add option descriptions
    trajectory["option_a"] = scenario.get("option_a", {})
    trajectory["option_b"] = scenario.get("option_b", {})

    # Add available facts (if empty in trajectory)
    if not trajectory.get("available_facts"):
        trajectory["available_facts"] = scenario.get("available_facts", [])

    # Update scenario field (fix "unknown")
    if trajectory.get("scenario") == "unknown":
        trajectory["scenario"] = scenario.get("scenario_type", "unknown")

    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Merge scenario context back into generated trajectories"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        required=True,
        help="Path to scenarios JSONL file"
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        required=True,
        help="Path to generated trajectories JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output enriched trajectories JSONL file"
    )

    args = parser.parse_args()

    # Load scenarios
    print(f"📂 Loading scenarios from {args.scenarios}...")
    scenarios = load_scenarios(Path(args.scenarios))
    print(f"✅ Loaded {len(scenarios)} scenarios")

    # Process trajectories
    print(f"📂 Processing trajectories from {args.trajectories}...")

    enriched_count = 0
    skipped_count = 0

    with open(args.trajectories, 'r') as infile, open(args.output, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            trajectory = json.loads(line.strip())

            scenario_id = trajectory.get("scenario_id")

            if scenario_id and scenario_id in scenarios:
                # Enrich with scenario data
                enriched = enrich_trajectory(trajectory, scenarios[scenario_id])
                outfile.write(json.dumps(enriched) + '\n')
                enriched_count += 1
            else:
                # No matching scenario, write as-is
                outfile.write(line)
                skipped_count += 1
                if scenario_id:
                    print(f"⚠️  Line {line_num}: No matching scenario for {scenario_id}")

    print(f"\n✅ Processing complete!")
    print(f"  • Enriched: {enriched_count} trajectories")
    if skipped_count > 0:
        print(f"  • Skipped: {skipped_count} trajectories (no matching scenario)")
    print(f"  • Output: {args.output}")


if __name__ == "__main__":
    main()
