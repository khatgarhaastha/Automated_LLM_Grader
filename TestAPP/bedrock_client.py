import boto3
import os

def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION")
    )

def invoke_model(prompt: str) -> str:
    client = get_bedrock_client()

    response = client.invoke_model(
        body=prompt.encode('utf-8'),
        modelId=os.getenv("MODEL_ID"),  # e.g., 'anthropic.claude-3-sonnet-20240229-v1:0'
        contentType='application/json',
        accept='application/json'
    )
    return response['body'].read().decode('utf-8')
