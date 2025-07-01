# debug_setup.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 DEBUGGING YOUR SETUP")
print("=" * 50)

# Check what's actually loaded
model_id = os.getenv("MODEL_ID")
region = os.getenv("AWS_REGION")
inference_arn = os.getenv("INFERENCE_PROFILE_ARN")

print(f"MODEL_ID: '{model_id}'")
print(f"Length of MODEL_ID: {len(model_id) if model_id else 'None'}")
print(f"AWS_REGION: '{region}'")
print(f"INFERENCE_PROFILE_ARN: '{inference_arn}'")

# Check for hidden characters
if model_id:
    print(f"MODEL_ID repr: {repr(model_id)}")
    print(f"MODEL_ID starts with space: {model_id.startswith(' ')}")
    print(f"MODEL_ID ends with space: {model_id.endswith(' ')}")

print("\n" + "=" * 50)

# Test what your bedrock client would use
model_identifier = inference_arn if inference_arn else model_id
print(f"🎯 MODEL IDENTIFIER BEING USED: '{model_identifier}'")
print(f"Model identifier repr: {repr(model_identifier)}")

# Check if .env file exists and show its contents
print(f"\n📁 .env file contents:")
try:
    with open('.env', 'r') as f:
        content = f.read()
        print(repr(content))
except FileNotFoundError:
    print("❌ .env file not found!")

print("\n🧪 Testing with different model IDs...")

# Test different model ID formats
test_ids = [
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0"  # Known working model
]

import boto3
client = boto3.client('bedrock-runtime', region_name=region)

for test_id in test_ids:
    print(f"Testing: {test_id}")
    try:
        # Just test if the model ID is valid (don't actually invoke)
        response = client.invoke_model(
            modelId=test_id,
            body='{"anthropic_version": "bedrock-2023-05-31", "max_tokens": 1, "messages": [{"role": "user", "content": "test"}]}',
            contentType="application/json",
            accept="application/json"
        )
        print(f"  ✅ VALID: {test_id}")
    except Exception as e:
        print(f"  ❌ INVALID: {test_id} - {str(e)[:100]}...")