#!/usr/bin/env python3
"""Test Together AI API response format."""

from together import Together
import os

client = Together(api_key="tgp_v1_hLwdeVS73vVZt5BCtHbQTQPCPFYDhoFo1NqMcqM9Dkc")

response = client.chat.completions.create(
    model="irawadee_5d65/Meta-Llama-3.1-8B-Instruct-Reference-ptom-agent-5713b704",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ],
    max_tokens=100,
)

print("Response type:", type(response))
print("Response:", response)
print("\nChoice type:", type(response.choices[0]))
print("Choice:", response.choices[0])
print("\nChoice attributes:", dir(response.choices[0]))

# Try to access content
try:
    print("\nTrying .message.content:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")

try:
    print("\nTrying .text:")
    print(response.choices[0].text)
except Exception as e:
    print(f"Error: {e}")
