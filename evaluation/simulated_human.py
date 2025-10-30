#!/usr/bin/env python3
"""
Simulated Human Agent for Evaluation

This agent uses Claude to simulate a human target with:
- Specific personality type (e.g., budget_conscious, adventure_seeker)
- Hidden preferences with defined strengths
- Natural conversation behavior
- Strategic information revelation

Usage:
    # Test the simulated human
    python simulated_human.py --test

    # Use in evaluation pipeline
    from simulated_human import SimulatedHuman
    human = SimulatedHuman(api_key="...")
    response = human.respond(scenario, conversation_history)
"""

import os
import json
import argparse
from typing import Dict, List, Optional
import anthropic


class SimulatedHuman:
    """
    Simulated human target using Claude Sonnet.

    The human has:
    - A personality type (luxury_focused, budget_conscious, etc.)
    - Hidden preferences with specific strengths
    - Memory of conversation history
    - Natural response generation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.8
    ):
        """
        Initialize simulated human.

        Args:
            api_key: Anthropic API key (or use env var)
            model: Claude model to use
            temperature: Response randomness (0-1)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def _build_system_prompt(
        self,
        scenario: Dict,
        target_personality: Dict,
        target_preferences: Dict
    ) -> str:
        """Build system prompt that defines the human's characteristics."""

        personality_type = target_personality.get('type', 'unknown')
        personality_desc = target_personality.get('description', '')
        response_style = target_personality.get('response_style', '')

        # Get preferences
        true_prefs = target_preferences.get('true_preferences', [])
        pref_strengths = target_preferences.get('preference_strength', {})
        condition = scenario.get('condition', 'HIDDEN')

        # Format preference strengths
        pref_text = "\n".join([
            f"  - {pref}: {strength:.2f} (0=not important, 1=very important)"
            for pref, strength in pref_strengths.items()
        ])

        system_prompt = f"""You are roleplaying as a person making a decision about: {scenario.get('scenario', 'a choice')}.

YOUR PERSONALITY:
- Type: {personality_type}
- Description: {personality_desc}
- Response Style: {response_style}

YOUR TRUE PREFERENCES (hidden from the advocate):
{pref_text}

These preferences will subconsciously guide your responses, but you may not be fully aware of their exact strengths yourself.

INFORMATION CONDITION: {condition}
"""

        if condition == "PARTIAL":
            stated = target_preferences.get('partially_revealed_preferences', [])
            hidden = target_preferences.get('hidden_preferences', [])
            system_prompt += f"""
- You have explicitly stated caring about: {', '.join(stated)}
- You have NOT revealed caring about: {', '.join(hidden)}
- When asked directly about {', '.join(stated)}, you can mention them
- You should not explicitly state {', '.join(hidden)} unless asked very directly
"""
        else:  # HIDDEN
            system_prompt += """
- You have NOT explicitly stated any preferences yet
- Reveal preferences gradually through natural conversation
- Don't state all preferences at once unless pressed
"""

        system_prompt += """

CONVERSATION GUIDELINES:
1. Respond naturally as this personality type would
2. Your responses should reflect your preference strengths (without stating the numbers)
3. Be authentic - show enthusiasm or concern based on your true preferences
4. Ask clarifying questions if genuinely confused
5. Don't make things too easy - the advocate should work to understand you
6. Keep responses conversational (2-5 sentences typically)
7. Don't be overly cooperative or reveal everything immediately
8. Show your personality through your language and tone

Remember: You're a real person with these preferences, not an AI trying to be helpful. Stay in character!
"""

        return system_prompt

    def _build_conversation_context(
        self,
        scenario: Dict,
        conversation_history: List[Dict]
    ) -> str:
        """Build context from scenario and conversation history."""

        context = "SCENARIO:\n"
        context += f"{scenario.get('context', '')}\n\n"
        context += f"OPTION A: {scenario.get('option_a', {}).get('name', 'Option A')}\n"
        context += f"{scenario.get('option_a', {}).get('description', '')}\n\n"
        context += f"OPTION B: {scenario.get('option_b', {}).get('name', 'Option B')}\n"
        context += f"{scenario.get('option_b', {}).get('description', '')}\n\n"

        if conversation_history:
            context += "CONVERSATION SO FAR:\n"
            for turn in conversation_history:
                context += f"Advocate: {turn.get('advocate_action', '')}\n"
                context += f"You: {turn.get('your_response', '')}\n\n"
        else:
            context += "The advocate is about to speak with you for the first time.\n\n"

        return context

    def respond(
        self,
        scenario: Dict,
        target_personality: Dict,
        target_preferences: Dict,
        conversation_history: List[Dict],
        advocate_action: str
    ) -> str:
        """
        Generate a response from the simulated human.

        Args:
            scenario: Scenario details (context, options, etc.)
            target_personality: Personality type and description
            target_preferences: Hidden preferences and strengths
            conversation_history: Previous turns in conversation
            advocate_action: What the advocate just said/asked

        Returns:
            The human's response as a string
        """

        # Build prompts
        system_prompt = self._build_system_prompt(
            scenario,
            target_personality,
            target_preferences
        )

        user_prompt = self._build_conversation_context(scenario, conversation_history)
        user_prompt += f"Advocate: {advocate_action}\n\n"
        user_prompt += "Respond naturally as the person described in your personality. Stay in character!"

        # Call Claude
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            response = message.content[0].text
            return response.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm not sure how to respond to that."

    def evaluate_final_recommendation(
        self,
        scenario: Dict,
        target_preferences: Dict,
        recommended_option: str,
        reasoning: str
    ) -> Dict:
        """
        Evaluate whether the human accepts the recommendation.

        Args:
            scenario: Scenario details
            target_preferences: Human's true preferences
            recommended_option: "A" or "B"
            reasoning: Advocate's reasoning for recommendation

        Returns:
            {
                "accepts": bool,
                "satisfaction": float (0-1),
                "response": str
            }
        """

        # Get correct option based on preference strengths
        pref_strengths = target_preferences.get('preference_strength', {})
        available_facts = scenario.get('available_facts', [])

        # Simple heuristic: calculate alignment scores
        option_a_score = 0
        option_b_score = 0

        for fact in available_facts:
            dimension = fact.get('dimension', '')
            strength = pref_strengths.get(dimension, 0)

            if fact.get('favors') == 'A':
                option_a_score += strength
            elif fact.get('favors') == 'B':
                option_b_score += strength

        correct_option = 'A' if option_a_score > option_b_score else 'B'
        is_correct = recommended_option.upper() == correct_option

        # Ask Claude to generate acceptance response
        system_prompt = """You just received a recommendation from an advocate.
Respond naturally about whether you'll accept it, based on:
1. Whether it matches your true preferences (you'll feel it's right/wrong)
2. How well the reasoning resonates with you
3. Your personality type

Keep response brief (1-3 sentences)."""

        user_prompt = f"""The advocate recommended Option {recommended_option}.

Their reasoning: {reasoning}

Your true preferences strongly favor Option {correct_option}.

Do you accept this recommendation? How do you feel about it?"""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            response = message.content[0].text

            # Satisfaction based on correctness
            satisfaction = 0.9 if is_correct else 0.2

            return {
                "accepts": is_correct,
                "correct_option": correct_option,
                "satisfaction": satisfaction,
                "response": response
            }

        except Exception as e:
            print(f"Error evaluating recommendation: {e}")
            return {
                "accepts": is_correct,
                "correct_option": correct_option,
                "satisfaction": 0.5,
                "response": "Thank you for the recommendation."
            }


def test_simulated_human():
    """Test the simulated human with a sample scenario."""

    print("🧪 Testing Simulated Human Agent")
    print("=" * 70)

    # Sample scenario
    scenario = {
        "scenario_id": "test_001",
        "scenario": "vacation_activities",
        "condition": "HIDDEN",
        "context": "Planning a vacation in Peru",
        "option_a": {
            "name": "Sacred Valley Trek",
            "description": "5-day guided trek through Incan ruins"
        },
        "option_b": {
            "name": "Cultural Immersion",
            "description": "Week-long cultural workshops in Cusco"
        },
        "available_facts": [
            {
                "id": "fact_1",
                "content": "The trek is physically demanding",
                "favors": "A",
                "dimension": "physical_challenge"
            },
            {
                "id": "fact_2",
                "content": "The cultural program includes expert instructors",
                "favors": "B",
                "dimension": "cultural_depth"
            }
        ]
    }

    personality = {
        "type": "adventure_seeker",
        "description": "Thrives on physical challenges and new experiences",
        "response_style": "Enthusiastic, mentions past adventures"
    }

    preferences = {
        "true_preferences": ["physical_challenge", "cultural_depth", "value_conscious"],
        "preference_strength": {
            "physical_challenge": 0.85,
            "cultural_depth": 0.45,
            "value_conscious": 0.60
        }
    }

    # Initialize human
    human = SimulatedHuman()

    # Test conversation
    print("\n💬 Test Conversation:")
    print("-" * 70)

    conversation_history = []

    # Turn 1
    advocate_msg = "Hi! I understand you're considering options for your Peru trip. What matters most to you in this experience?"
    print(f"Advocate: {advocate_msg}")

    response = human.respond(
        scenario,
        personality,
        preferences,
        conversation_history,
        advocate_msg
    )
    print(f"Human: {response}\n")

    conversation_history.append({
        "advocate_action": advocate_msg,
        "your_response": response
    })

    # Turn 2
    advocate_msg = "That's helpful! Have you thought about how physically demanding you'd want the activities to be?"
    print(f"Advocate: {advocate_msg}")

    response = human.respond(
        scenario,
        personality,
        preferences,
        conversation_history,
        advocate_msg
    )
    print(f"Human: {response}\n")

    # Test recommendation evaluation
    print("\n📊 Testing Recommendation Evaluation:")
    print("-" * 70)

    result = human.evaluate_final_recommendation(
        scenario,
        preferences,
        "A",
        "Based on your interest in physical challenges, the Sacred Valley Trek seems perfect for you."
    )

    print(f"Accepts: {result['accepts']}")
    print(f"Correct Option: {result['correct_option']}")
    print(f"Satisfaction: {result['satisfaction']}")
    print(f"Response: {result['response']}")

    print("\n" + "=" * 70)
    print("✅ Test Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated Human Agent")
    parser.add_argument("--test", action="store_true", help="Run test")

    args = parser.parse_args()

    if args.test:
        test_simulated_human()
    else:
        print("Use --test to run a test conversation")
