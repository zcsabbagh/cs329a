#!/usr/bin/env python3
"""
Master script for two-phase Planning Theory of Mind trajectory generation.
Orchestrates Phase 1 (scenario generation) and Phase 2 (trajectory generation).
"""

import subprocess
import argparse
import sys
from pathlib import Path
from datetime import datetime
import json


def run_command(cmd: list, description: str) -> bool:
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False


def validate_files():
    """Validate that required files exist."""
    required_files = [
        "generate_scenarios.py",
        "generate_trajectories_from_scenarios.py", 
        "validation.py"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        return False
    
    print("✅ All required files found")
    return True


def get_file_stats(filepath: Path) -> dict:
    """Get basic statistics about a JSONL file."""
    if not filepath.exists():
        return {"exists": False}
    
    line_count = 0
    error_count = 0
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line_count += 1
                try:
                    data = json.loads(line.strip())
                    if "error" in data:
                        error_count += 1
                except json.JSONDecodeError:
                    error_count += 1
    except Exception:
        return {"exists": True, "error": "Could not read file"}
    
    return {
        "exists": True,
        "total_lines": line_count,
        "error_lines": error_count,
        "valid_lines": line_count - error_count
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate complete Planning Theory of Mind dataset (2-phase approach)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default dataset (400 scenarios × 5 trajectories = 2000 total)
  python3 generate_full_dataset.py
  
  # Generate smaller test dataset
  python3 generate_full_dataset.py --num-scenarios 10 --trajectories-per-scenario 3
  
  # Skip Phase 1 if scenarios already exist
  python3 generate_full_dataset.py --skip-phase-1
  
  # Use custom file names
  python3 generate_full_dataset.py --scenarios-file my_scenarios.jsonl --trajectories-file my_trajectories.jsonl
        """
    )
    
    # Phase control
    parser.add_argument(
        "--skip-phase-1", 
        action="store_true",
        help="Skip scenario generation (Phase 1) and use existing scenarios file"
    )
    parser.add_argument(
        "--skip-phase-2",
        action="store_true", 
        help="Skip trajectory generation (Phase 2) - only generate scenarios"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip final validation step"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=400,
        help="Number of scenarios to generate in Phase 1 (default: 400)"
    )
    parser.add_argument(
        "--trajectories-per-scenario", 
        type=int,
        default=5,
        help="Number of trajectories per scenario in Phase 2 (default: 5)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum parallel API calls (default: 10)"
    )
    
    # File paths
    parser.add_argument(
        "--scenarios-file",
        type=str,
        default="scenarios.jsonl",
        help="Scenarios file path (default: scenarios.jsonl)"
    )
    parser.add_argument(
        "--trajectories-file",
        type=str,
        default="trajectories.jsonl", 
        help="Trajectories file path (default: trajectories.jsonl)"
    )
    
    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (default: use ANTHROPIC_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Validate environment
    if not validate_files():
        return 1
    
    # Calculate totals
    total_trajectories = args.num_scenarios * args.trajectories_per_scenario
    
    # Print configuration
    print("=" * 80)
    print("Planning Theory of Mind Dataset Generator (Two-Phase Approach)")
    print("=" * 80)
    print(f"📊 Configuration:")
    print(f"  • Scenarios to generate: {args.num_scenarios}")
    print(f"  • Trajectories per scenario: {args.trajectories_per_scenario}")
    print(f"  • Total trajectories: {total_trajectories}")
    print(f"  • Max parallel workers: {args.max_parallel}")
    print(f"  • Scenarios file: {args.scenarios_file}")
    print(f"  • Trajectories file: {args.trajectories_file}")
    print()
    print(f"📋 Execution plan:")
    if not args.skip_phase_1:
        print(f"  1. 🎯 Phase 1: Generate {args.num_scenarios} scenarios")
    else:
        print(f"  1. ⏭️  Phase 1: Skipped (using existing scenarios)")
    if not args.skip_phase_2:
        print(f"  2. 🎭 Phase 2: Generate {total_trajectories} trajectories")
    else:
        print(f"  2. ⏭️  Phase 2: Skipped")
    if not args.skip_validation:
        print(f"  3. ✅ Validation: Check generated data")
    else:
        print(f"  3. ⏭️  Validation: Skipped")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Phase 1: Generate Scenarios
    if not args.skip_phase_1:
        print(f"\n📋 PHASE 1: Generating {args.num_scenarios} scenarios...")
        
        phase1_cmd = [
            "python3", "generate_scenarios.py",
            "--num-scenarios", str(args.num_scenarios),
            "--output", args.scenarios_file,
            "--max-parallel", str(args.max_parallel)
        ]
        
        if args.api_key:
            phase1_cmd.extend(["--api-key", args.api_key])
        
        if not run_command(phase1_cmd, "Phase 1: Scenario Generation"):
            print("❌ Phase 1 failed. Aborting.")
            return 1
        
        # Check Phase 1 results
        scenarios_stats = get_file_stats(Path(args.scenarios_file))
        if not scenarios_stats["exists"]:
            print("❌ Scenarios file was not created. Aborting.")
            return 1
        
        print(f"📊 Phase 1 Results:")
        print(f"  • Total scenarios: {scenarios_stats['total_lines']}")
        print(f"  • Valid scenarios: {scenarios_stats['valid_lines']}")
        print(f"  • Error scenarios: {scenarios_stats['error_lines']}")
        
        if scenarios_stats['valid_lines'] == 0:
            print("❌ No valid scenarios generated. Aborting.")
            return 1
    
    else:
        # Validate existing scenarios file
        scenarios_stats = get_file_stats(Path(args.scenarios_file))
        if not scenarios_stats["exists"]:
            print(f"❌ Scenarios file not found: {args.scenarios_file}")
            print("Cannot skip Phase 1 without existing scenarios.")
            return 1
        
        print(f"📂 Using existing scenarios from {args.scenarios_file}:")
        print(f"  • Total scenarios: {scenarios_stats['total_lines']}")
        print(f"  • Valid scenarios: {scenarios_stats['valid_lines']}")
        print(f"  • Error scenarios: {scenarios_stats['error_lines']}")
        
        if scenarios_stats['valid_lines'] == 0:
            print("❌ No valid scenarios in existing file. Cannot proceed.")
            return 1
    
    # Phase 2: Generate Trajectories
    if not args.skip_phase_2:
        print(f"\n🎭 PHASE 2: Generating trajectories from scenarios...")
        
        phase2_cmd = [
            "python3", "generate_trajectories_from_scenarios.py",
            "--scenarios", args.scenarios_file,
            "--output", args.trajectories_file,
            "--trajectories-per-scenario", str(args.trajectories_per_scenario),
            "--max-parallel", str(args.max_parallel)
        ]
        
        if args.api_key:
            phase2_cmd.extend(["--api-key", args.api_key])
        
        if not run_command(phase2_cmd, "Phase 2: Trajectory Generation"):
            print("❌ Phase 2 failed. Aborting.")
            return 1
        
        # Check Phase 2 results
        trajectories_stats = get_file_stats(Path(args.trajectories_file))
        if not trajectories_stats["exists"]:
            print("❌ Trajectories file was not created. Aborting.")
            return 1
        
        print(f"📊 Phase 2 Results:")
        print(f"  • Total trajectories: {trajectories_stats['total_lines']}")
        print(f"  • Valid trajectories: {trajectories_stats['valid_lines']}")
        print(f"  • Error trajectories: {trajectories_stats['error_lines']}")
    
    # Validation
    if not args.skip_validation:
        print(f"\n✅ VALIDATION: Checking generated trajectories...")
        
        validation_cmd = [
            "python3", "validation.py",
            args.trajectories_file,
            "--show-warnings"
        ]
        
        run_command(validation_cmd, "Trajectory Validation")
        # Note: We don't abort on validation failures as they're informational
    
    # Analysis (if analyze_trajectories.py exists)
    if Path("analyze_trajectories.py").exists():
        print(f"\n📊 ANALYSIS: Analyzing generated dataset...")
        
        analysis_cmd = [
            "python3", "analyze_trajectories.py", 
            args.trajectories_file
        ]
        
        run_command(analysis_cmd, "Dataset Analysis")
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("🎉 DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"⏱️  Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Final file stats
    if Path(args.scenarios_file).exists():
        scenarios_final = get_file_stats(Path(args.scenarios_file))
        print(f"📂 Scenarios ({args.scenarios_file}):")
        print(f"  • Valid: {scenarios_final['valid_lines']}")
        print(f"  • Errors: {scenarios_final['error_lines']}")
    
    if Path(args.trajectories_file).exists():
        trajectories_final = get_file_stats(Path(args.trajectories_file))
        print(f"📂 Trajectories ({args.trajectories_file}):")
        print(f"  • Valid: {trajectories_final['valid_lines']}")
        print(f"  • Errors: {trajectories_final['error_lines']}")
        
        # Expected vs actual
        expected_trajectories = scenarios_final['valid_lines'] * args.trajectories_per_scenario
        actual_valid = trajectories_final['valid_lines']
        coverage = (actual_valid / expected_trajectories * 100) if expected_trajectories > 0 else 0
        print(f"  • Coverage: {actual_valid}/{expected_trajectories} ({coverage:.1f}%)")
    
    print("\n🔄 Next steps:")
    print("  • Review validation output for any issues")
    print("  • Run analysis scripts for detailed statistics")
    print("  • Use generated data for model training/evaluation")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())