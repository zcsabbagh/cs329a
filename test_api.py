#!/usr/bin/env python3
"""Test the Anthropic API call to debug the response_format issue."""

import anthropic
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_api():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    
    try:
        # Test basic call
        message = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Generate a JSON object with just two fields: message and timestamp"
                }
            ]
        )
        print("✅ Basic API call successful!")
        print(f"Response: {message.content[0].text}")
        
        # Test with response_format (this should fail)
        print("\nTesting with response_format...")
        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": "Say hello"
                }
            ]
        )
        print("Response format test passed (unexpected)")
        
    except TypeError as e:
        print(f"❌ TypeError as expected: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_api())