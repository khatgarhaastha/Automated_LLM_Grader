from TestAPP.prompts import create_prompt
from TestAPP.bedrock_client import invoke_model

def evaluate_response(question, sample_answer, examples, target_answer):
    prompt = create_prompt(question, sample_answer, examples, target_answer)
    return invoke_model(prompt)
