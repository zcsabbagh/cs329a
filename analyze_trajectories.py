#!/usr/bin/env python3
"""
Enhanced analysis script for Planning Theory of Mind trajectories.
Provides comprehensive statistics and insights for both single-phase and two-phase datasets.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
import statistics


def load_trajectories(file_path: Path) -> List[Dict]:
    """Load trajectories from JSONL file."""
    trajectories = []
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                trajectory = json.loads(line.strip())
                trajectories.append(trajectory)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping line {line_num}: Invalid JSON - {e}")
                continue
    
    return trajectories


def analyze_basic_stats(trajectories: List[Dict]) -> Dict[str, Any]:
    """Analyze basic trajectory statistics."""
    total = len(trajectories)
    valid = sum(1 for t in trajectories if "error" not in t)
    errors = sum(1 for t in trajectories if "error" in t)
    
    # Success rate
    successful = sum(1 for t in trajectories if t.get("success", False) and "error" not in t)
    success_rate = (successful / valid * 100) if valid > 0 else 0
    
    # Turn counts
    turn_counts = []
    for t in trajectories:
        if "error" not in t and "trajectory" in t:
            turn_counts.append(len(t["trajectory"]))
    
    avg_turns = statistics.mean(turn_counts) if turn_counts else 0
    median_turns = statistics.median(turn_counts) if turn_counts else 0
    
    return {
        "total_trajectories": total,
        "valid_trajectories": valid,
        "error_trajectories": errors,
        "successful_trajectories": successful,
        "success_rate": success_rate,
        "avg_turns": avg_turns,
        "median_turns": median_turns,
        "turn_distribution": Counter(turn_counts)
    }


def analyze_by_scenario(trajectories: List[Dict]) -> Dict[str, Any]:
    """Analyze trajectory diversity per scenario (for two-phase datasets)."""
    by_scenario = defaultdict(list)
    scenario_coverage = {}
    
    for t in trajectories:
        if "error" not in t:
            scenario_id = t.get("scenario_id", "unknown")
            by_scenario[scenario_id].append(t)
    
    if not by_scenario:
        return {"has_scenarios": False}
    
    # Calculate statistics per scenario
    for scenario_id, scenario_trajectories in by_scenario.items():
        strategies = set(t.get("strategy_type", "unknown") for t in scenario_trajectories)
        success_count = sum(1 for t in scenario_trajectories if t.get("success", False))
        
        scenario_coverage[scenario_id] = {
            "total_trajectories": len(scenario_trajectories),
            "unique_strategies": len(strategies),
            "strategies": list(strategies),
            "success_count": success_count,
            "success_rate": (success_count / len(scenario_trajectories) * 100) if scenario_trajectories else 0
        }
    
    # Overall scenario statistics
    total_scenarios = len(by_scenario)
    avg_trajectories_per_scenario = sum(len(trajs) for trajs in by_scenario.values()) / total_scenarios
    
    # Scenarios with diverse strategies (4+ different strategies)
    diverse_scenarios = sum(1 for info in scenario_coverage.values() if info["unique_strategies"] >= 4)
    
    return {
        "has_scenarios": True,
        "total_scenarios": total_scenarios,
        "avg_trajectories_per_scenario": avg_trajectories_per_scenario,
        "diverse_scenarios": diverse_scenarios,
        "diverse_scenario_percentage": (diverse_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0,
        "scenario_details": scenario_coverage
    }


def analyze_strategies(trajectories: List[Dict]) -> Dict[str, Any]:
    """Analyze strategy types and their performance."""
    strategy_stats = defaultdict(lambda: {
        "total": 0,
        "successful": 0,
        "avg_turns": [],
        "validation_failed": 0
    })
    
    for t in trajectories:
        if "error" not in t:
            strategy = t.get("strategy_type", "unknown")
            strategy_stats[strategy]["total"] += 1
            
            if t.get("success", False):
                strategy_stats[strategy]["successful"] += 1
            
            if t.get("_validation_failed", False):
                strategy_stats[strategy]["validation_failed"] += 1
            
            if "trajectory" in t:
                strategy_stats[strategy]["avg_turns"].append(len(t["trajectory"]))
    
    # Calculate averages
    strategy_summary = {}
    for strategy, stats in strategy_stats.items():
        success_rate = (stats["successful"] / stats["total"] * 100) if stats["total"] > 0 else 0
        avg_turns = statistics.mean(stats["avg_turns"]) if stats["avg_turns"] else 0
        
        strategy_summary[strategy] = {
            "total": stats["total"],
            "successful": stats["successful"],
            "success_rate": success_rate,
            "avg_turns": avg_turns,
            "validation_failed": stats["validation_failed"]
        }
    
    return strategy_summary


def analyze_counterfactuals(trajectories: List[Dict]) -> Dict[str, Any]:
    """Analyze counterfactual coverage and quality."""
    total_turns = 0
    turns_with_counterfactuals = 0
    counterfactual_alternatives = []
    
    for t in trajectories:
        if "error" not in t and "trajectory" in t:
            for turn in t["trajectory"]:
                if isinstance(turn, dict):
                    total_turns += 1
                    
                    if "counterfactual_analysis" in turn:
                        turns_with_counterfactuals += 1
                        cf_analysis = turn["counterfactual_analysis"]
                        
                        if isinstance(cf_analysis, dict):
                            counterfactual_alternatives.append(len(cf_analysis))
    
    coverage_percentage = (turns_with_counterfactuals / total_turns * 100) if total_turns > 0 else 0
    avg_alternatives = statistics.mean(counterfactual_alternatives) if counterfactual_alternatives else 0
    
    return {
        "total_turns": total_turns,
        "turns_with_counterfactuals": turns_with_counterfactuals,
        "coverage_percentage": coverage_percentage,
        "avg_alternatives_per_counterfactual": avg_alternatives,
        "counterfactual_distribution": Counter(counterfactual_alternatives)
    }


def analyze_scenario_types(trajectories: List[Dict]) -> Dict[str, Any]:
    """Analyze distribution of scenario types."""
    scenario_types = Counter()
    conditions = Counter()
    complexities = Counter()
    
    for t in trajectories:
        if "error" not in t:
            scenario_types[t.get("scenario", "unknown")] += 1
            conditions[t.get("condition", "unknown")] += 1
            
            # Check for complexity in base scenario
            if "_base_scenario" in t:
                complexities[t["_base_scenario"].get("complexity", "unknown")] += 1
    
    return {
        "scenario_types": dict(scenario_types),
        "conditions": dict(conditions),
        "complexities": dict(complexities) if any(complexities.values()) else {}
    }


def analyze_validation_quality(trajectories: List[Dict]) -> Dict[str, Any]:
    """Analyze validation attempts and quality metrics."""
    validation_attempts = []
    validation_failed = 0
    counterfactual_coverage = []
    
    for t in trajectories:
        if "error" not in t:
            attempts = t.get("_validation_attempts", 1)
            validation_attempts.append(attempts)
            
            if t.get("_validation_failed", False):
                validation_failed += 1
            
            coverage = t.get("_counterfactual_coverage", 0)
            if coverage > 0:
                counterfactual_coverage.append(coverage)
    
    avg_attempts = statistics.mean(validation_attempts) if validation_attempts else 0
    avg_coverage = statistics.mean(counterfactual_coverage) if counterfactual_coverage else 0
    
    return {
        "avg_validation_attempts": avg_attempts,
        "validation_failed_count": validation_failed,
        "avg_counterfactual_coverage": avg_coverage,
        "validation_attempt_distribution": Counter(validation_attempts)
    }


def print_analysis_report(trajectories: List[Dict], file_path: Path):
    """Print comprehensive analysis report."""
    print("=" * 80)
    print(f"📊 PLANNING THEORY OF MIND TRAJECTORY ANALYSIS")
    print("=" * 80)
    print(f"📂 File: {file_path}")
    print(f"📅 Analysis time: {Path().cwd()}")
    print()
    
    # Basic Statistics
    basic_stats = analyze_basic_stats(trajectories)
    print("📈 BASIC STATISTICS")
    print("-" * 40)
    print(f"  • Total trajectories: {basic_stats['total_trajectories']:,}")
    print(f"  • Valid trajectories: {basic_stats['valid_trajectories']:,}")
    print(f"  • Error trajectories: {basic_stats['error_trajectories']:,}")
    print(f"  • Success rate: {basic_stats['success_rate']:.1f}% ({basic_stats['successful_trajectories']}/{basic_stats['valid_trajectories']})")
    print(f"  • Average turns: {basic_stats['avg_turns']:.1f}")
    print(f"  • Median turns: {basic_stats['median_turns']:.1f}")
    
    if basic_stats['turn_distribution']:
        print(f"  • Turn distribution:")
        for turns, count in sorted(basic_stats['turn_distribution'].items()):
            percentage = (count / basic_stats['valid_trajectories'] * 100) if basic_stats['valid_trajectories'] > 0 else 0
            print(f"    - {turns} turns: {count:,} ({percentage:.1f}%)")
    print()
    
    # Scenario Analysis (for two-phase datasets)
    scenario_analysis = analyze_by_scenario(trajectories)
    if scenario_analysis["has_scenarios"]:
        print("🎯 SCENARIO ANALYSIS")
        print("-" * 40)
        print(f"  • Unique scenarios: {scenario_analysis['total_scenarios']:,}")
        print(f"  • Avg trajectories per scenario: {scenario_analysis['avg_trajectories_per_scenario']:.1f}")
        print(f"  • Scenarios with 4+ strategies: {scenario_analysis['diverse_scenarios']} ({scenario_analysis['diverse_scenario_percentage']:.1f}%)")
        
        # Show top scenarios by trajectory count
        scenario_details = scenario_analysis['scenario_details']
        top_scenarios = sorted(scenario_details.items(), key=lambda x: x[1]['total_trajectories'], reverse=True)[:5]
        
        print(f"  • Top scenarios by trajectory count:")
        for scenario_id, details in top_scenarios:
            print(f"    - {scenario_id}: {details['total_trajectories']} trajectories, {details['unique_strategies']} strategies")
        print()
    
    # Strategy Analysis
    strategy_stats = analyze_strategies(trajectories)
    if strategy_stats:
        print("🎭 STRATEGY ANALYSIS")
        print("-" * 40)
        for strategy, stats in sorted(strategy_stats.items()):
            print(f"  • {strategy}:")
            print(f"    - Total: {stats['total']:,}")
            print(f"    - Success rate: {stats['success_rate']:.1f}% ({stats['successful']}/{stats['total']})")
            print(f"    - Avg turns: {stats['avg_turns']:.1f}")
            if stats['validation_failed'] > 0:
                print(f"    - Validation failed: {stats['validation_failed']}")
        print()
    
    # Counterfactual Analysis
    cf_analysis = analyze_counterfactuals(trajectories)
    print("🔮 COUNTERFACTUAL ANALYSIS")
    print("-" * 40)
    print(f"  • Total decision turns: {cf_analysis['total_turns']:,}")
    print(f"  • Turns with counterfactuals: {cf_analysis['turns_with_counterfactuals']:,}")
    print(f"  • Coverage: {cf_analysis['coverage_percentage']:.1f}%")
    print(f"  • Avg alternatives per counterfactual: {cf_analysis['avg_alternatives_per_counterfactual']:.1f}")
    
    if cf_analysis['counterfactual_distribution']:
        print(f"  • Alternatives distribution:")
        for alt_count, frequency in sorted(cf_analysis['counterfactual_distribution'].items()):
            print(f"    - {alt_count} alternatives: {frequency:,} counterfactuals")
    print()
    
    # Scenario Types
    type_analysis = analyze_scenario_types(trajectories)
    print("📋 SCENARIO TYPES")
    print("-" * 40)
    print(f"  • Scenario types:")
    for scenario_type, count in sorted(type_analysis['scenario_types'].items()):
        percentage = (count / basic_stats['valid_trajectories'] * 100) if basic_stats['valid_trajectories'] > 0 else 0
        print(f"    - {scenario_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"  • Conditions:")
    for condition, count in sorted(type_analysis['conditions'].items()):
        percentage = (count / basic_stats['valid_trajectories'] * 100) if basic_stats['valid_trajectories'] > 0 else 0
        print(f"    - {condition}: {count:,} ({percentage:.1f}%)")
    
    if type_analysis['complexities']:
        print(f"  • Complexity levels:")
        for complexity, count in sorted(type_analysis['complexities'].items()):
            percentage = (count / basic_stats['valid_trajectories'] * 100) if basic_stats['valid_trajectories'] > 0 else 0
            print(f"    - {complexity}: {count:,} ({percentage:.1f}%)")
    print()
    
    # Validation Quality
    validation_analysis = analyze_validation_quality(trajectories)
    print("✅ VALIDATION QUALITY")
    print("-" * 40)
    print(f"  • Avg validation attempts: {validation_analysis['avg_validation_attempts']:.1f}")
    print(f"  • Validation failures: {validation_analysis['validation_failed_count']:,}")
    print(f"  • Avg counterfactual coverage: {validation_analysis['avg_counterfactual_coverage']:.1%}")
    
    if validation_analysis['validation_attempt_distribution']:
        print(f"  • Validation attempts distribution:")
        for attempts, count in sorted(validation_analysis['validation_attempt_distribution'].items()):
            print(f"    - {attempts} attempts: {count:,} trajectories")
    print()
    
    # Quality Assessment
    print("🏆 QUALITY ASSESSMENT")
    print("-" * 40)
    
    # Overall quality score
    quality_factors = []
    
    # Success rate factor (0-1)
    quality_factors.append(min(basic_stats['success_rate'] / 80.0, 1.0))  # Target 80% success
    
    # Counterfactual coverage factor (0-1) 
    quality_factors.append(min(cf_analysis['coverage_percentage'] / 80.0, 1.0))  # Target 80% coverage
    
    # Strategy diversity factor (0-1) - for two-phase datasets
    if scenario_analysis["has_scenarios"] and scenario_analysis['total_scenarios'] > 0:
        quality_factors.append(min(scenario_analysis['diverse_scenario_percentage'] / 80.0, 1.0))  # Target 80% diverse
    
    # Validation quality factor (0-1)
    quality_factors.append(1.0 - min(validation_analysis['validation_failed_count'] / basic_stats['valid_trajectories'], 0.2))
    
    overall_quality = statistics.mean(quality_factors) * 100
    
    print(f"  • Overall Quality Score: {overall_quality:.1f}/100")
    
    if overall_quality >= 90:
        quality_label = "Excellent 🌟"
    elif overall_quality >= 80:
        quality_label = "Good ✅"
    elif overall_quality >= 70:
        quality_label = "Fair ⚠️"
    else:
        quality_label = "Needs Improvement ❌"
    
    print(f"  • Quality Rating: {quality_label}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    if basic_stats['success_rate'] < 75:
        print(f"  • Consider improving success rate (currently {basic_stats['success_rate']:.1f}%, target 75%+)")
    
    if cf_analysis['coverage_percentage'] < 80:
        print(f"  • Increase counterfactual coverage (currently {cf_analysis['coverage_percentage']:.1f}%, target 80%+)")
    
    if scenario_analysis["has_scenarios"] and scenario_analysis['diverse_scenario_percentage'] < 80:
        print(f"  • Increase strategy diversity per scenario (currently {scenario_analysis['diverse_scenario_percentage']:.1f}%, target 80%+)")
    
    if validation_analysis['validation_failed_count'] > basic_stats['valid_trajectories'] * 0.1:
        print(f"  • Reduce validation failures (currently {validation_analysis['validation_failed_count']} failures)")
    
    if not any([
        basic_stats['success_rate'] < 75,
        cf_analysis['coverage_percentage'] < 80,
        scenario_analysis["has_scenarios"] and scenario_analysis['diverse_scenario_percentage'] < 80,
        validation_analysis['validation_failed_count'] > basic_stats['valid_trajectories'] * 0.1
    ]):
        print(f"  • Dataset quality is excellent! No major issues detected.")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Planning Theory of Mind trajectory datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze trajectory dataset
  python3 analyze_trajectories.py trajectories.jsonl
  
  # Analyze with JSON output
  python3 analyze_trajectories.py trajectories.jsonl --json-output analysis.json
        """
    )
    
    parser.add_argument(
        "trajectories_file",
        type=str,
        help="Path to trajectories JSONL file"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        help="Save analysis results to JSON file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printed output (useful with --json-output)"
    )
    
    args = parser.parse_args()
    
    # Load trajectories
    trajectories_file = Path(args.trajectories_file)
    try:
        trajectories = load_trajectories(trajectories_file)
    except Exception as e:
        print(f"❌ Error loading trajectories: {e}")
        return 1
    
    if len(trajectories) == 0:
        print("❌ No trajectories found in file")
        return 1
    
    # Print analysis report
    if not args.quiet:
        print_analysis_report(trajectories, trajectories_file)
    
    # Save JSON output if requested
    if args.json_output:
        analysis_results = {
            "basic_stats": analyze_basic_stats(trajectories),
            "scenario_analysis": analyze_by_scenario(trajectories),
            "strategy_analysis": analyze_strategies(trajectories),
            "counterfactual_analysis": analyze_counterfactuals(trajectories),
            "scenario_types": analyze_scenario_types(trajectories),
            "validation_quality": analyze_validation_quality(trajectories)
        }
        
        json_output_path = Path(args.json_output)
        with open(json_output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        if not args.quiet:
            print(f"📄 Analysis results saved to: {json_output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())