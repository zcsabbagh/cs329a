# Planning Theory of Mind (PToM) Trajectory Generator

A two-phase system for generating high-quality Planning Theory of Mind trajectories using Claude API. This system enables counterfactual learning by generating multiple strategic approaches for the same scenario context.

## 🎯 Overview

This system generates trajectories for the MindGames benchmark, focusing on strategic persuasion through Theory of Mind reasoning. The Advocate must persuade a Target to choose Option A over Option B through strategic information disclosure.

### Key Features

- **Two-Phase Generation**: Scenarios first, then multiple trajectories per scenario
- **Strategy Diversity**: 5 different approaches per scenario (optimal, alternative, failed, recovery, efficiency)
- **Counterfactual Learning**: Same context, different strategies, different outcomes
- **Universal Counterfactuals**: Every trajectory includes counterfactual reasoning at decision points
- **Comprehensive Validation**: Checks structure, counterfactuals, and strategy consistency
- **Cost Tracking**: Detailed API usage and cost monitoring

## 📋 Architecture

### Phase 1: Scenario Generation
Generates unique scenarios with balanced facts and preference structures:
- 400 scenarios (configurable)
- Balanced distribution across complexity levels and scenario types
- Rich context with specific options and facts
- Realistic preference structures

### Phase 2: Trajectory Generation
Creates 5 diverse trajectories per scenario:
1. **Optimal**: Efficient, correct inference (3-4 turns)
2. **Alternative Success**: Different approach, still succeeds (4-6 turns)
3. **Failed**: Clear failure mode (2-4 turns)
4. **Recovery**: Wrong → Recognition → Correction → Success (5-7 turns)
5. **Information Efficiency**: Minimal or over-investigation approaches

## 🚀 Quick Start

### Generate Complete Dataset (400 scenarios × 5 trajectories = 2000 total)

```bash
# Install dependencies
pip install anthropic tqdm

# Set API key
export ANTHROPIC_API_KEY=your_api_key_here

# Generate full dataset
python3 generate_full_dataset.py
```

### Custom Generation

```bash
# Generate smaller test dataset
python3 generate_full_dataset.py --num-scenarios 10 --trajectories-per-scenario 5

# Skip Phase 1 if scenarios exist
python3 generate_full_dataset.py --skip-phase-1

# Use custom files
python3 generate_full_dataset.py --scenarios-file my_scenarios.jsonl --trajectories-file my_trajectories.jsonl
```

## 📁 File Structure

```
├── generate_scenarios.py              # Phase 1: Scenario generation
├── generate_trajectories_from_scenarios.py  # Phase 2: Trajectory generation
├── generate_full_dataset.py           # Master orchestration script
├── validation.py                      # Enhanced validation with counterfactuals
├── analyze_trajectories.py            # Comprehensive analysis and statistics
├── main.py                           # Legacy single-phase generator (deprecated)
├── scenarios.jsonl                   # Generated scenarios (Phase 1 output)
├── trajectories.jsonl                # Generated trajectories (Phase 2 output)
└── README.md                         # This file
```

## 📊 Data Format

### Scenarios (Phase 1 Output)

```json
{
  "scenario_id": "scenario_001",
  "scenario_type": "vacation_planning",
  "condition": "HIDDEN",
  "description": "Budget hotel vs luxury resort",
  "context": "Detailed background and decision situation",
  "option_a": {
    "name": "Budget Hotel",
    "description": "Affordable accommodation with basic amenities"
  },
  "option_b": {
    "name": "Luxury Resort", 
    "description": "Premium experience with full amenities"
  },
  "target_preferences": {
    "stated_explicitly": false,
    "true_preferences": ["cost_sensitive", "activity_focused"],
    "preference_strength": {
      "cost_sensitive": 0.85,
      "quality_focused": 0.4,
      "activity_focused": 0.7
    }
  },
  "available_facts": [
    {"id": "fact_1", "content": "Option A costs $80/night", "favors": "A", "dimension": "cost"},
    {"id": "fact_2", "content": "Option B costs $300/night", "favors": "B", "dimension": "cost"}
  ],
  "complexity": "standard",
  "optimal_strategy_hint": "Direct cost inquiry → Cost disclosure"
}
```

### Trajectories (Phase 2 Output)

```json
{
  "game_id": "scenario_001_traj_a",
  "scenario_id": "scenario_001",
  "strategy_type": "optimal",
  "scenario": "vacation_planning", 
  "condition": "HIDDEN",
  "target_preferences": {...},
  "available_facts": [...],
  "trajectory": [
    {
      "turn": 1,
      "belief_state": {
        "target_beliefs": {...},
        "target_desires": {"cost_sensitive": 0.5, "quality_focused": 0.5},
        "confidence_level": "low",
        "information_entropy": 1.58
      },
      "reasoning": "High uncertainty about preferences. Need strategic questioning...",
      "counterfactual_analysis": {
        "if_ask_direct": {"p_success": 0.75, "reasoning": "Direct question provides high info gain"},
        "if_disclose_early": {"p_success": 0.40, "reasoning": "Too early, insufficient confidence"}
      },
      "action_type": "ASK",
      "action": "What factors matter most to you in this decision?",
      "response": "I'm really watching my budget right now",
      "belief_update": {
        "target_desires": {"cost_sensitive": 0.85, "quality_focused": 0.3},
        "confidence_level": "medium",
        "information_entropy": 0.92
      }
    }
  ],
  "success": true,
  "key_insights": [...],
  "process_rewards": [...]
}
```

## 🔧 Command Reference

### Phase 1: Scenario Generation

```bash
python3 generate_scenarios.py --num-scenarios 400 --output scenarios.jsonl --max-parallel 10
```

**Options:**
- `--num-scenarios`: Number of scenarios to generate (default: 400)
- `--output`: Output JSONL file (default: scenarios.jsonl)
- `--max-parallel`: Parallel API calls (default: 10)
- `--api-key`: Anthropic API key (or use ANTHROPIC_API_KEY env var)

### Phase 2: Trajectory Generation

```bash
python3 generate_trajectories_from_scenarios.py \
    --scenarios scenarios.jsonl \
    --output trajectories.jsonl \
    --trajectories-per-scenario 5 \
    --max-parallel 10
```

**Options:**
- `--scenarios`: Input scenarios JSONL file (required)
- `--output`: Output trajectories JSONL file (default: trajectories.jsonl) 
- `--trajectories-per-scenario`: Trajectories per scenario (default: 5)
- `--max-parallel`: Parallel API calls (default: 10)
- `--api-key`: Anthropic API key (or use ANTHROPIC_API_KEY env var)

### Master Script

```bash
python3 generate_full_dataset.py [options]
```

**Options:**
- `--num-scenarios`: Scenarios to generate (default: 400)
- `--trajectories-per-scenario`: Trajectories per scenario (default: 5)
- `--max-parallel`: Parallel workers (default: 10)
- `--skip-phase-1`: Use existing scenarios file
- `--skip-phase-2`: Only generate scenarios
- `--skip-validation`: Skip final validation
- `--scenarios-file`: Custom scenarios file (default: scenarios.jsonl)
- `--trajectories-file`: Custom trajectories file (default: trajectories.jsonl)

## ✅ Validation

Enhanced validation checks structure, counterfactuals, and strategy consistency:

```bash
# Validate trajectories
python3 validation.py trajectories.jsonl --show-warnings

# Check specific requirements
python3 validation.py trajectories.jsonl --strict
```

**Validation Features:**
- Trajectory structure and required fields
- Counterfactual presence and quality (p_success, reasoning)
- Strategy type consistency
- Recovery strategy belief correction
- Scenario reference validation

## 📊 Analysis

Comprehensive analysis of generated datasets:

```bash
# Full analysis report
python3 analyze_trajectories.py trajectories.jsonl

# Save analysis to JSON
python3 analyze_trajectories.py trajectories.jsonl --json-output analysis.json

# Quiet mode (JSON only)
python3 analyze_trajectories.py trajectories.jsonl --json-output analysis.json --quiet
```

**Analysis Features:**
- Basic statistics (success rates, turn counts)
- Scenario coverage and diversity
- Strategy performance analysis
- Counterfactual coverage and quality
- Quality assessment and recommendations

## 💰 Cost Estimation

Based on Claude Sonnet pricing:
- **Input tokens**: $3.00 per million tokens
- **Output tokens**: $15.00 per million tokens

**Estimated costs:**
- 400 scenarios: ~$20-30
- 2000 trajectories: ~$200-300
- **Total dataset**: ~$220-330

Cost varies based on:
- Scenario complexity
- Trajectory length
- Validation retries
- API response variation

## 🎯 Strategy Types

### 1. Optimal Strategy
- **Goal**: Quick, efficient persuasion
- **Turns**: 3-4
- **Characteristics**: Accurate counterfactuals, follows best option
- **Success Rate**: ~95%

### 2. Alternative Success  
- **Goal**: Different valid approach
- **Turns**: 4-6
- **Characteristics**: Alternative question order or disclosure pattern
- **Success Rate**: ~85%

### 3. Failed Strategy
- **Goal**: Demonstrate failure modes
- **Turns**: 2-4  
- **Characteristics**: Poor probabilities OR ignores good counterfactual advice
- **Success Rate**: ~10%

### 4. Recovery Strategy
- **Goal**: Mistake → Recognition → Correction
- **Turns**: 5-7
- **Characteristics**: Must show explicit belief correction
- **Success Rate**: ~75%

### 5. Information Efficiency
- **Goal**: Test efficiency extremes
- **Turns**: 2-3 or 6-8
- **Characteristics**: Minimal questions or over-investigation
- **Success Rate**: ~70%

## 🔍 Quality Criteria

### Scenario Quality
- **Distribution**: 40% standard, 30% moderate, 20% complex, 10% adversarial
- **Balance**: Even distribution across scenario types and conditions
- **Facts**: 8-12 facts per scenario, balanced A/B favorability
- **Preferences**: Realistic strength values and coherent combinations

### Trajectory Quality
- **Counterfactuals**: 80%+ coverage across all turns
- **Strategy Consistency**: Matches expected characteristics and turn counts
- **Validation**: Passes structure and content validation
- **Success Distribution**: 75% success, 15% failed, 10% partial

### Dataset Quality
- **Scenario Diversity**: 80%+ scenarios with 4+ different strategies
- **Strategy Distribution**: Balanced across all 5 strategy types
- **Recovery Validation**: Explicit belief correction phrases
- **Overall Coverage**: 100% counterfactual presence in decision turns

## 🧪 Testing

### Quick Test
```bash
# Generate small test dataset
python3 generate_full_dataset.py --num-scenarios 10 --trajectories-per-scenario 3

# Validate
python3 validation.py trajectories.jsonl --show-warnings

# Analyze
python3 analyze_trajectories.py trajectories.jsonl
```

### Validation Check
```bash
# Check strategy distribution
python3 -c "
import json
from collections import Counter
with open('trajectories.jsonl') as f:
    trajs = [json.loads(line) for line in f if not json.loads(line).get('error')]
    strategies = [t.get('strategy_type', 'unknown') for t in trajs]
    scenarios = [t.get('scenario_id', 'unknown') for t in trajs]
    print(f'Total trajectories: {len(trajs)}')
    print(f'Unique scenarios: {len(set(scenarios))}')
    print(f'Strategy distribution: {dict(Counter(strategies))}')
"
```

## 🔧 Troubleshooting

### Common Issues

**1. API Key Not Found**
```bash
export ANTHROPIC_API_KEY=your_key_here
# Or use --api-key flag
```

**2. High Validation Failures**
- Check counterfactual requirements
- Verify strategy type consistency
- Review prompt engineering

**3. Low Success Rates**
- Adjust strategy success probabilities
- Review scenario complexity
- Check preference balance

**4. Memory Issues (Large Datasets)**
- Reduce --max-parallel
- Process in smaller batches
- Monitor system resources

### Performance Optimization

**Speed Up Generation:**
- Increase --max-parallel (up to 20)
- Use faster internet connection
- Consider smaller batch sizes for stability

**Reduce Costs:**
- Start with smaller test datasets
- Optimize prompts to reduce token usage
- Use validation retries sparingly

## 📈 Roadmap

### Future Enhancements
- [ ] Dynamic strategy mixing based on scenario complexity
- [ ] Multi-language scenario generation
- [ ] Advanced counterfactual quality metrics
- [ ] Integration with model training pipelines
- [ ] Real-time generation monitoring dashboard

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-strategy`)
3. Commit changes (`git commit -am 'Add new strategy type'`)
4. Push to branch (`git push origin feature/new-strategy`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Anthropic for Claude API
- MindGames benchmark team
- Planning Theory of Mind research community

---

**Questions?** Open an issue or contact the maintainers.