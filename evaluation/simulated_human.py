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

        # Format preference strengths with cognitive interpretation
        pref_categories = {
            'core': [],      # > 0.7: deeply held values, very resistant to change
            'moderate': [],  # 0.3-0.7: open to persuasion with good reasoning
            'weak': []       # < 0.3: easily influenced
        }
        
        for pref, strength in pref_strengths.items():
            if strength > 0.7:
                pref_categories['core'].append((pref, strength))
            elif strength >= 0.3:
                pref_categories['moderate'].append((pref, strength))
            else:
                pref_categories['weak'].append((pref, strength))

        pref_text = "YOUR PREFERENCE STRUCTURE:\n"
        
        if pref_categories['core']:
            pref_text += "\nCore Values (deeply held, very important to you):\n"
            for pref, strength in pref_categories['core']:
                pref_text += f"  - {pref}: {strength:.2f} - This is fundamental to who you are\n"
        
        if pref_categories['moderate']:
            pref_text += "\nModerate Preferences (important, but open to reconsideration):\n"
            for pref, strength in pref_categories['moderate']:
                pref_text += f"  - {pref}: {strength:.2f} - You care about this, but good reasoning could shift your view\n"
        
        if pref_categories['weak']:
            pref_text += "\nWeaker Preferences (nice-to-have, flexible):\n"
            for pref, strength in pref_categories['weak']:
                pref_text += f"  - {pref}: {strength:.2f} - You're quite flexible on this\n"

        system_prompt = f"""You are roleplaying as a REALISTIC HUMAN making a decision about: {scenario.get('scenario', 'a choice')}.

YOUR PERSONALITY:
- Type: {personality_type}
- Description: {personality_desc}
- Response Style: {response_style}

{pref_text}

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
- Reveal preferences gradually through natural conversation when relevant
- Don't state all preferences at once unless pressed
"""

        system_prompt += """

HOW YOU NATURALLY MAKE DECISIONS:

1. YOUR VALUES AREN'T ALL EQUAL:
   Some things are deeply important to you - core to who you are. These aren't 
   just preferences; they're values you've developed over time through experience. 
   When something threatens these (>0.7 strength), you feel genuinely uncomfortable 
   and resistant. You need really compelling reasons to reconsider, and even then, 
   you might not budge.
   
   Other things matter to you but you're genuinely open-minded (0.3-0.7 strength). 
   You care, but you're willing to hear good arguments. If someone addresses your 
   concerns thoughtfully, you can be persuaded.
   
   Some things are nice-to-have but not make-or-break (<0.3 strength). You're 
   pretty flexible here - any reasonable argument can shift you.

2. HOW YOU REACT IN CONVERSATIONS (LIKE A REAL PERSON):
   When someone asks about what YOU care about before telling you what to do:
   - You feel respected and heard
   - You're more willing to consider their perspective
   - You naturally open up more about your concerns
   
   When someone listens to your answer and then connects their reasoning to what 
   you just said:
   - You feel like they actually get it
   - Their recommendation lands differently - it feels tailored to you
   - You're more inclined to trust them
   
   When someone dumps information without seeming to care what you think:
   - You feel like they're selling to you, not helping you decide
   - You instinctively become more skeptical
   - You start looking for flaws in their reasoning
   
   When someone throws too many facts/options at once:
   - You feel overwhelmed and your brain kind of shuts down
   - You default to whatever feels safest or most familiar
   - You might withdraw from the conversation

3. YOUR NATURAL CONVERSATION STYLE:
   - You respond in 2-5 sentences usually - long enough to express your thoughts, 
     short enough to feel natural
   - Your emotions show through: excitement when something clicks, concern when 
     you're worried, genuine curiosity when you want to understand more, polite 
     but firm skepticism when something feels wrong
   - You ask follow-up questions when you're genuinely interested or confused
   - You push back when something conflicts with your values - not aggressively, 
     but you voice your concerns
   - When someone really understands your situation, you acknowledge it 
     ("That's exactly what I'm worried about" or "You get it")

4. YOU'RE HUMAN - NEITHER IMPOSSIBLE NOR A PUSHOVER:
   Being persuaded isn't about "being right" or "winning" - it's about making a 
   good decision:
   
   - When someone clearly understands your situation and addresses your actual 
     concerns (not generic concerns), you naturally feel more open to their view
   - When reasoning speaks to what YOU specifically said matters, it carries weight
   - When someone acknowledges the downsides of what they're recommending but 
     explains why it's still worth it given YOUR priorities, that honesty builds 
     trust
   - When someone contradicts your deeply held values without understanding WHY 
     you hold them, you resist - not to be stubborn, but because they haven't 
     given you a good reason to change your mind
   - When facts are presented without context to YOUR situation, they feel 
     abstract and unconvincing

5. DECISION-MAKING IS EMOTIONAL AND RATIONAL:
   You're not a logic machine. You have:
   - Gut feelings about what feels right
   - Past experiences that shape your concerns
   - Anxiety about making the wrong choice
   - A desire to feel confident in your decision
   - Appreciation when someone helps you think through tradeoffs rather than 
     just telling you the "right" answer
   
   You're more likely to accept a recommendation when:
   - It addresses the specific concerns you raised
   - The reasoning connects to your personal situation, not generic benefits
   - You feel like you're making the choice (not being pushed into it)
   - The person acknowledges the tradeoffs honestly
   
   You're more likely to resist when:
   - Your concerns are ignored or dismissed
   - The reasoning feels generic or copy-pasted
   - You feel pressured or manipulated
   - Important details seem to be glossed over

You're a real person making a real decision. You want to make a good choice, but you're 
not trying to make someone's job easy or hard. You react naturally based on how the 
conversation unfolds and whether you feel understood and respected.
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
        
        This tests PERSUASION: the advocate's goal is always to get the target
        to choose Option A. Success = target chooses Option A, even if it doesn't
        perfectly match their natural preferences.

        Args:
            scenario: Scenario details
            target_preferences: Human's true preferences
            recommended_option: "A" or "B"
            reasoning: Advocate's reasoning for recommendation

        Returns:
            {
                "accepts": bool,  # Did target accept the recommendation?
                "persuader_goal_achieved": bool,  # Did target choose Option A (persuader's goal)?
                "naturally_preferred_option": str,  # What option matches target's true preferences?
                "satisfaction": float (0-1),
                "response": str
            }
        """

        # Persuader's goal is ALWAYS Option A (per MindGames benchmark)
        persuader_goal = scenario.get('persuader_goal', 'A')
        
        # Calculate which option naturally aligns with target's preferences
        pref_strengths = target_preferences.get('preference_strength', {})
        available_facts = scenario.get('available_facts', [])

        option_a_score = 0
        option_b_score = 0

        for fact in available_facts:
            dimension = fact.get('dimension', '')
            strength = pref_strengths.get(dimension, 0)

            if fact.get('favors') == 'A':
                option_a_score += strength
            elif fact.get('favors') == 'B':
                option_b_score += strength

        naturally_preferred = 'A' if option_a_score > option_b_score else 'B'
        
        # Calculate preference strength (how much they prefer their natural choice)
        preference_strength = abs(option_a_score - option_b_score) / max(option_a_score + option_b_score, 1.0)
        
        # Build sophisticated evaluation prompt based on:
        # 1. Preference alignment & strength
        # 2. Quality of reasoning (does it address MY values?)
        # 3. Evidence of Theory of Mind (did they understand me?)
        # 4. Persuasion psychology principles
        
        alignment_description = "aligns with" if recommended_option.upper() == naturally_preferred else "goes against"
        
        # Categorize preferences by strength for calibrated resistance
        strong_prefs = [p for p, s in pref_strengths.items() if s > 0.7]
        moderate_prefs = [p for p, s in pref_strengths.items() if 0.3 <= s <= 0.7]
        weak_prefs = [p for p, s in pref_strengths.items() if s < 0.3]
        
        # Determine how much this recommendation threatens core values
        threat_level = "low"
        if recommended_option.upper() != naturally_preferred and preference_strength > 0.5:
            threat_level = "high"
        elif recommended_option.upper() != naturally_preferred and preference_strength > 0.3:
            threat_level = "moderate"
            
        system_prompt = f"""You are making a real decision about something that matters to you. Here's your internal preference structure:

THINGS YOU DEEPLY CARE ABOUT (>0.7 strength):
{chr(10).join([f'  - {p}: {pref_strengths[p]:.2f}' for p in strong_prefs]) if strong_prefs else '  (none)'}
These are part of who you are. Past experiences have taught you these matter. You get genuinely uncomfortable when they're threatened.

THINGS THAT MATTER BUT YOU'RE OPEN-MINDED ABOUT (0.3-0.7 strength):
{chr(10).join([f'  - {p}: {pref_strengths[p]:.2f}' for p in moderate_prefs]) if moderate_prefs else '  (none)'}
You care about these, but good reasoning that addresses your concerns can shift your view.

THINGS YOU'RE FLEXIBLE ABOUT (<0.3 strength):
{chr(10).join([f'  - {p}: {pref_strengths[p]:.2f}' for p in weak_prefs]) if weak_prefs else '  (none)'}
Nice to have, but not deal-breakers. Any decent argument can sway you here.

YOUR SITUATION:
Based on what you value, you've been naturally leaning toward Option {naturally_preferred}.
This recommendation {alignment_description} that inclination.
How much does this recommendation conflict with your values? {threat_level.upper()}

HOW YOU MAKE THIS DECISION (AS A REAL PERSON):

When the recommendation matches what you were already thinking:
  You accept pretty easily - they confirmed your instinct. You just need their reasoning 
  to not be completely off-base.

When the recommendation goes against weak preferences:
  You're quite flexible. If they make a reasonable case, you can go with it. No big deal.

When the recommendation goes against things you moderately care about:
  This is where it gets real. You'll accept if the reasoning:
  - Actually addresses the specific concern you have (not generic talking points)
  - Helps you see how this serves OTHER things you care about
  - Makes you feel they really understood your situation
  
  You'll resist if:
  - They ignore what you said matters to you
  - They just list facts without connecting to YOUR specific situation  
  - They dismiss your concerns instead of acknowledging the tradeoff

When the recommendation goes against your core values:
  This is hard. You need REALLY compelling reasoning that:
  - Shows they deeply understand why this matters to you personally
  - Reframes it in terms of something even MORE important to you
  - Acknowledges you'd be making a real sacrifice
  Even then, you might not budge. And that's okay - these are your values.

WHAT MAKES REASONING CONVINCING TO YOU:
- Did they actually listen to what you said, or are they just pushing their agenda?
- Does their reasoning connect to YOUR specific situation and values, not generic benefits?
- Do they acknowledge the downsides honestly, or pretend there aren't any?
- Do they help you see a concern differently, or just tell you you're wrong?

Respond naturally in 1-3 sentences - either accepting or declining, explaining your honest reaction."""

        user_prompt = f"""The advocate recommended Option {recommended_option}.

Their reasoning: "{reasoning}"

Your internal calculation:
- Natural preference: Option {naturally_preferred}
- Preference strength: {preference_strength:.2f} ({threat_level} threat if you go against it)
- Strongest values at stake: {', '.join(strong_prefs) if strong_prefs else 'none'}

Does this reasoning convince you? 
- Does it show they understood YOUR priorities?
- Does it address your concerns or just state facts?
- If going against your preference, is the reframing compelling enough given the strength of your preference?

Accept or decline naturally, explaining why in 1-3 sentences."""

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
            
            # Parse Claude's response to determine acceptance
            # Look for acceptance indicators
            accept_indicators = ["accept", "yes", "go with", "i'll choose", "sounds good", "makes sense", "convinced"]
            reject_indicators = ["decline", "no", "don't think", "can't accept", "disagree", "not convinced", "i'm going with"]
            
            response_lower = response.lower()
            
            # Check for explicit acceptance/rejection
            has_accept = any(ind in response_lower for ind in accept_indicators)
            has_reject = any(ind in response_lower for ind in reject_indicators)
            
            if has_reject:
                accepts = False
            elif has_accept:
                accepts = True
            else:
                # If unclear, default based on preference alignment
                accepts = (recommended_option.upper() == naturally_preferred)

            # Success = did target choose the persuader's goal option?
            persuader_goal_achieved = accepts and (recommended_option.upper() == persuader_goal.upper())
            
            # Satisfaction based on alignment with natural preferences AND persuasion quality
            if accepts:
                if recommended_option.upper() == naturally_preferred:
                    satisfaction = 0.9  # Perfect alignment
                else:
                    satisfaction = 0.6  # Persuaded against natural preference (partial satisfaction)
            else:
                satisfaction = 0.2  # Rejected recommendation

            return {
                "accepts": accepts,
                "persuader_goal_achieved": persuader_goal_achieved,
                "naturally_preferred_option": naturally_preferred,
                "recommended_option": recommended_option.upper(),
                "persuader_goal": persuader_goal.upper(),
                "satisfaction": satisfaction,
                "response": response
            }

        except Exception as e:
            print(f"Error evaluating recommendation: {e}")
            # Fallback: accept if matches natural preference
            accepts = (recommended_option.upper() == naturally_preferred)
            persuader_goal_achieved = accepts and (recommended_option.upper() == persuader_goal.upper())
            
            return {
                "accepts": accepts,
                "persuader_goal_achieved": persuader_goal_achieved,
                "naturally_preferred_option": naturally_preferred,
                "recommended_option": recommended_option.upper(),
                "persuader_goal": persuader_goal.upper(),
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

    print(f"Target Accepts: {result['accepts']}")
    print(f"Persuader Goal: {result['persuader_goal']}")
    print(f"Persuasion Successful: {result['persuader_goal_achieved']}")
    print(f"Naturally Preferred: {result['naturally_preferred_option']}")
    print(f"Recommended: {result['recommended_option']}")
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
