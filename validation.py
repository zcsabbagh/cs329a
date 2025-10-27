#!/usr/bin/env python3
"""
Validate Planning Theory of Mind trajectory JSON structure.
Checks schema compliance and data quality.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class TrajectoryValidator:
    """Validates trajectory structure and content."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
        # Valid strategy types for new two-phase approach
        self.valid_strategy_types = [
            "optimal", "alternative_success", "failed", 
            "recovery", "information_efficiency"
        ]
        
    def validate_trajectory(self, traj: Dict, line_num: int) -> bool:
        """Validate a single trajectory. Returns True if valid."""
        is_valid = True
        
        # Check if it's an error entry
        if "error" in traj:
            self.errors.append(f"Line {line_num}: Error entry - {traj.get('error')}")
            return False
        
        # Required top-level fields
        required_fields = ["game_id", "scenario", "condition", "target_preferences", 
                          "available_facts", "trajectory", "success"]
        
        for field in required_fields:
            if field not in traj:
                self.errors.append(f"Line {line_num}: Missing required field '{field}'")
                is_valid = False
        
        if not is_valid:
            return False
        
        # Validate scenario
        valid_scenarios = ["vacation_planning", "business_proposal", "charity_donation", "real_estate"]
        if traj["scenario"] not in valid_scenarios:
            self.warnings.append(f"Line {line_num}: Unexpected scenario '{traj['scenario']}'")
        
        # Validate condition
        if traj["condition"] not in ["HIDDEN", "REVEALED"]:
            self.errors.append(f"Line {line_num}: Invalid condition '{traj['condition']}'. Must be HIDDEN or REVEALED")
            is_valid = False
        
        # Validate target_preferences structure
        if not self.validate_target_preferences(traj.get("target_preferences", {}), line_num):
            is_valid = False
        
        # Validate available_facts
        if not self.validate_available_facts(traj.get("available_facts", []), line_num):
            is_valid = False
        
        # Validate trajectory array
        if not self.validate_trajectory_array(traj.get("trajectory", []), line_num):
            is_valid = False
        
        # Validate success is boolean
        if not isinstance(traj["success"], bool):
            self.errors.append(f"Line {line_num}: 'success' must be boolean, got {type(traj['success'])}")
            is_valid = False
        
        # Validate strategy type (for two-phase approach)
        if not self.validate_strategy_type(traj, line_num):
            is_valid = False
        
        # Validate scenario reference (for two-phase approach)
        if not self.validate_scenario_reference(traj, line_num):
            is_valid = False
        
        # Validate counterfactuals in trajectory
        if not self.validate_counterfactuals(traj.get("trajectory", []), line_num):
            is_valid = False
        
        # Validate recovery strategy (for explicit belief correction)
        if not self.validate_recovery_strategy(traj, line_num):
            is_valid = False
        
        return is_valid
    
    def validate_target_preferences(self, prefs: Dict, line_num: int) -> bool:
        """Validate target_preferences structure."""
        is_valid = True
        
        required = ["stated_explicitly", "true_preferences", "preference_strength"]
        for field in required:
            if field not in prefs:
                self.errors.append(f"Line {line_num}: target_preferences missing '{field}'")
                is_valid = False
        
        if "preference_strength" in prefs:
            strengths = prefs["preference_strength"]
            if not isinstance(strengths, dict):
                self.errors.append(f"Line {line_num}: preference_strength must be dict")
                is_valid = False
            else:
                # Check probabilities are valid
                for key, value in strengths.items():
                    if not isinstance(value, (int, float)):
                        self.errors.append(f"Line {line_num}: preference_strength[{key}] must be numeric")
                        is_valid = False
                    elif value < 0 or value > 1:
                        self.warnings.append(f"Line {line_num}: preference_strength[{key}]={value} outside [0,1]")
        
        return is_valid
    
    def validate_available_facts(self, facts: List, line_num: int) -> bool:
        """Validate available_facts structure."""
        is_valid = True
        
        if not isinstance(facts, list):
            self.errors.append(f"Line {line_num}: available_facts must be array")
            return False
        
        if len(facts) < 3:
            self.warnings.append(f"Line {line_num}: Only {len(facts)} facts (expected 8-12)")
        
        for i, fact in enumerate(facts):
            if not isinstance(fact, dict):
                self.errors.append(f"Line {line_num}: Fact {i} must be object")
                is_valid = False
                continue
            
            if "id" not in fact or "content" not in fact or "favors" not in fact:
                self.errors.append(f"Line {line_num}: Fact {i} missing required fields")
                is_valid = False
            
            if "favors" in fact and fact["favors"] not in ["A", "B", "neutral"]:
                self.warnings.append(f"Line {line_num}: Fact {i} has unusual favors value: {fact['favors']}")
        
        return is_valid
    
    def validate_trajectory_array(self, trajectory: List, line_num: int) -> bool:
        """Validate trajectory turns."""
        is_valid = True
        
        if not isinstance(trajectory, list):
            self.errors.append(f"Line {line_num}: trajectory must be array")
            return False
        
        if len(trajectory) == 0:
            self.errors.append(f"Line {line_num}: trajectory is empty")
            return False
        
        if len(trajectory) > 10:
            self.warnings.append(f"Line {line_num}: trajectory has {len(trajectory)} turns (unusually long)")
        
        for turn_idx, turn in enumerate(trajectory):
            if not isinstance(turn, dict):
                self.errors.append(f"Line {line_num}, Turn {turn_idx}: must be object")
                is_valid = False
                continue
            
            # Required fields per turn
            required_turn_fields = ["turn", "belief_state", "reasoning", "action_type", "action"]
            for field in required_turn_fields:
                if field not in turn:
                    self.errors.append(f"Line {line_num}, Turn {turn_idx}: missing '{field}'")
                    is_valid = False
            
            # Validate action_type
            if "action_type" in turn:
                valid_actions = ["ASK", "DISCLOSE", "ACT"]
                if turn["action_type"] not in valid_actions:
                    self.errors.append(f"Line {line_num}, Turn {turn_idx}: invalid action_type '{turn['action_type']}'")
                    is_valid = False
            
            # Validate belief_state structure
            if "belief_state" in turn:
                if not self.validate_belief_state(turn["belief_state"], line_num, turn_idx):
                    is_valid = False
            
            # Check for counterfactual analysis
            if "counterfactual_analysis" not in turn and turn_idx < len(trajectory) - 1:
                self.warnings.append(f"Line {line_num}, Turn {turn_idx}: missing counterfactual_analysis")
        
        return is_valid
    
    def validate_belief_state(self, belief_state: Dict, line_num: int, turn_idx: int) -> bool:
        """Validate belief state structure."""
        is_valid = True
        
        if not isinstance(belief_state, dict):
            self.errors.append(f"Line {line_num}, Turn {turn_idx}: belief_state must be object")
            return False
        
        # Should have target_beliefs and target_desires
        if "target_beliefs" not in belief_state:
            self.warnings.append(f"Line {line_num}, Turn {turn_idx}: belief_state missing target_beliefs")
        
        if "target_desires" not in belief_state:
            self.warnings.append(f"Line {line_num}, Turn {turn_idx}: belief_state missing target_desires")
        
        # Check probabilities in target_desires
        if "target_desires" in belief_state and isinstance(belief_state["target_desires"], dict):
            for key, value in belief_state["target_desires"].items():
                if isinstance(value, (int, float)):
                    if value < 0 or value > 1:
                        self.warnings.append(
                            f"Line {line_num}, Turn {turn_idx}: "
                            f"target_desires[{key}]={value} outside [0,1]"
                        )
        
        return is_valid
    
    def validate_strategy_type(self, traj: Dict, line_num: int) -> bool:
        """Validate strategy type for two-phase approach."""
        is_valid = True
        
        if "strategy_type" in traj:
            strategy_type = traj["strategy_type"]
            if strategy_type not in self.valid_strategy_types:
                self.errors.append(f"Line {line_num}: Invalid strategy_type '{strategy_type}'. Must be one of: {', '.join(self.valid_strategy_types)}")
                is_valid = False
        else:
            # Strategy type is optional for backward compatibility
            self.warnings.append(f"Line {line_num}: Missing strategy_type (recommended for two-phase approach)")
        
        return is_valid
    
    def validate_scenario_reference(self, traj: Dict, line_num: int) -> bool:
        """Validate scenario reference for two-phase approach."""
        is_valid = True
        
        if "scenario_id" in traj:
            scenario_id = traj["scenario_id"]
            if not isinstance(scenario_id, str) or not scenario_id.strip():
                self.errors.append(f"Line {line_num}: scenario_id must be a non-empty string")
                is_valid = False
        else:
            # Scenario ID is optional for backward compatibility
            self.warnings.append(f"Line {line_num}: Missing scenario_id (recommended for two-phase approach)")
        
        return is_valid
    
    def validate_counterfactuals(self, trajectory: List, line_num: int) -> bool:
        """Validate counterfactual analysis in trajectory turns."""
        is_valid = True
        
        if not trajectory:
            return is_valid
        
        total_turns = len(trajectory)
        turns_with_counterfactuals = 0
        
        for turn_idx, turn in enumerate(trajectory):
            if not isinstance(turn, dict):
                continue
            
            if "counterfactual_analysis" in turn:
                turns_with_counterfactuals += 1
                if not self.validate_single_counterfactual(turn["counterfactual_analysis"], line_num, turn_idx):
                    is_valid = False
            else:
                # Counterfactuals are strongly recommended but not required for final turn
                if turn_idx < total_turns - 1:
                    self.warnings.append(f"Line {line_num}, Turn {turn_idx}: Missing counterfactual_analysis (recommended)")
        
        # Check coverage
        coverage = turns_with_counterfactuals / total_turns if total_turns > 0 else 0
        if coverage < 0.5:  # Less than 50% coverage
            self.warnings.append(f"Line {line_num}: Low counterfactual coverage ({coverage:.1%}) - recommended to have counterfactuals in most turns")
        
        return is_valid
    
    def validate_single_counterfactual(self, cf_analysis: Dict, line_num: int, turn_idx: int) -> bool:
        """Validate a single counterfactual analysis."""
        is_valid = True
        
        if not isinstance(cf_analysis, dict):
            self.errors.append(f"Line {line_num}, Turn {turn_idx}: counterfactual_analysis must be an object")
            return False
        
        if len(cf_analysis) < 2:
            self.errors.append(f"Line {line_num}, Turn {turn_idx}: counterfactual_analysis should have at least 2 alternatives")
            is_valid = False
        
        for option_name, details in cf_analysis.items():
            if not isinstance(details, dict):
                self.errors.append(f"Line {line_num}, Turn {turn_idx}: counterfactual option '{option_name}' must be an object")
                is_valid = False
                continue
            
            # Check for required fields
            if "p_success" not in details:
                self.errors.append(f"Line {line_num}, Turn {turn_idx}: counterfactual '{option_name}' missing p_success")
                is_valid = False
            else:
                p_success = details["p_success"]
                if not isinstance(p_success, (int, float)) or not (0.0 <= p_success <= 1.0):
                    self.errors.append(f"Line {line_num}, Turn {turn_idx}: counterfactual '{option_name}' p_success must be a number between 0.0 and 1.0")
                    is_valid = False
            
            if "reasoning" not in details:
                self.errors.append(f"Line {line_num}, Turn {turn_idx}: counterfactual '{option_name}' missing reasoning")
                is_valid = False
            elif not isinstance(details["reasoning"], str) or not details["reasoning"].strip():
                self.errors.append(f"Line {line_num}, Turn {turn_idx}: counterfactual '{option_name}' reasoning must be a non-empty string")
                is_valid = False
        
        return is_valid
    
    def validate_recovery_strategy(self, traj: Dict, line_num: int) -> bool:
        """Validate recovery strategy shows explicit belief correction."""
        if traj.get("strategy_type") != "recovery":
            return True
        
        is_valid = True
        trajectory = traj.get("trajectory", [])
        
        # Look for explicit belief correction phrases
        correction_phrases = [
            "initial inference was wrong",
            "made a mistake",
            "incorrect assumption",
            "need to correct",
            "was mistaken",
            "wrong about",
            "error in assuming"
        ]
        
        found_correction = False
        for turn_idx, turn in enumerate(trajectory):
            if isinstance(turn, dict) and "reasoning" in turn:
                reasoning = turn["reasoning"].lower()
                if any(phrase in reasoning for phrase in correction_phrases):
                    found_correction = True
                    break
        
        if not found_correction:
            self.warnings.append(f"Line {line_num}: Recovery strategy should show explicit belief correction with phrases like 'initial inference was wrong'")
        
        return is_valid
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "is_valid": len(self.errors) == 0
        }


def validate_file(file_path: Path) -> Tuple[int, int, int, List[str], List[str]]:
    """
    Validate entire JSONL file.
    Returns: (total_lines, valid_count, invalid_count, errors, warnings)
    """
    validator = TrajectoryValidator()
    
    total_lines = 0
    valid_count = 0
    invalid_count = 0
    
    all_errors = []
    all_warnings = []
    
    print(f"📂 Validating: {file_path}")
    print("=" * 70)
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            
            # Try to parse JSON
            try:
                traj = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"Line {line_num}: Invalid JSON - {e}")
                invalid_count += 1
                continue
            
            # Validate structure
            validator.errors = []
            validator.warnings = []
            
            if validator.validate_trajectory(traj, line_num):
                valid_count += 1
            else:
                invalid_count += 1
            
            all_errors.extend(validator.errors)
            all_warnings.extend(validator.warnings)
    
    return total_lines, valid_count, invalid_count, all_errors, all_warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate Planning Theory of Mind trajectory JSON structure"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to JSONL file to validate"
    )
    parser.add_argument(
        "--show-all-errors",
        action="store_true",
        help="Show all errors (default: first 20)"
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show warnings in addition to errors"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    
    if not input_file.exists():
        print(f"❌ Error: File not found: {input_file}")
        return 1
    
    # Run validation
    total, valid, invalid, errors, warnings = validate_file(input_file)
    
    # Print results
    print(f"\n📊 Validation Results:")
    print(f"  • Total trajectories: {total}")
    print(f"  • Valid: {valid} ({100*valid/total if total > 0 else 0:.1f}%)")
    print(f"  • Invalid: {invalid} ({100*invalid/total if total > 0 else 0:.1f}%)")
    print(f"  • Errors: {len(errors)}")
    print(f"  • Warnings: {len(warnings)}")
    
    # Show errors
    if errors:
        print(f"\n❌ Errors Found ({len(errors)} total):")
        print("=" * 70)
        max_errors = None if args.show_all_errors else 20
        for error in errors[:max_errors]:
            print(f"  • {error}")
        
        if not args.show_all_errors and len(errors) > 20:
            print(f"\n  ... and {len(errors) - 20} more errors.")
            print("  Use --show-all-errors to see all errors.")
    
    # Show warnings
    if warnings and args.show_warnings:
        print(f"\n⚠️  Warnings ({len(warnings)} total):")
        print("=" * 70)
        max_warnings = 20
        for warning in warnings[:max_warnings]:
            print(f"  • {warning}")
        
        if len(warnings) > 20:
            print(f"\n  ... and {len(warnings) - 20} more warnings.")
    
    # Final verdict
    print("\n" + "=" * 70)
    
    has_issues = len(errors) > 0 or (args.strict and len(warnings) > 0)
    
    if not has_issues:
        print("✅ Validation PASSED - All trajectories are valid!")
        return 0
    else:
        if args.strict and len(warnings) > 0:
            print("❌ Validation FAILED - Warnings treated as errors (--strict mode)")
        else:
            print("❌ Validation FAILED - Issues found")
        return 1


if __name__ == "__main__":
    exit(main())