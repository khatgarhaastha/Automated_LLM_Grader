# bedrock_client.py
import boto3
import os
import json
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")
REGION = os.getenv("AWS_REGION")

def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION
    )

def invoke_model(prompt: str) -> str:
    client = get_bedrock_client()
    
    print(f"🔍 Using MODEL_ID: {MODEL_ID}")
    print(f"🔍 Using REGION: {REGION}")
    
    # Claude models use Messages API format
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "temperature": 0.3,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        # Use MODEL_ID directly (ignore inference profiles)
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
        
    except Exception as e:
        print(f"Error invoking model: {str(e)}")
        return f"Error: {str(e)}"