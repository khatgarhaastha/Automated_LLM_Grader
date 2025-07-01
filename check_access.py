# check_access.py
import boto3
import json

region = "us-east-1"
client = boto3.client('bedrock-runtime', region_name=region)

# Models from your earlier list
test_models = [
    "anthropic.claude-instant-v1",
    "anthropic.claude-v2",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0"
]

print("🔍 Testing model access...")
print("=" * 60)

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 10,
    "temperature": 0.1,
    "messages": [{"role": "user", "content": "test"}]
}

working_models = []

for model_id in test_models:
    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        print(f"✅ WORKS: {model_id}")
        working_models.append(model_id)
    except Exception as e:
        error_type = "AccessDenied" if "AccessDenied" in str(e) else "Other"
        print(f"❌ {error_type}: {model_id}")

print("\n" + "=" * 60)
print("🎯 WORKING MODELS:")
for model in working_models:
    print(f"   {model}")

if working_models:
    print(f"\n💡 Use this in your .env file:")
    print(f"MODEL_ID={working_models[0]}")
else:
    print("\n❌ No models are accessible. Check Bedrock model access permissions.")