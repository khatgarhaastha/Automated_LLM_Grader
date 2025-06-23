import json
from TestAPP.evaluator import evaluate_response
from TestStorage.store import save_result

# Load input data
with open("data/sample_input.json") as f:
    payload = json.load(f)

question = payload["question"]
sample_answer = payload["sample_answer"]
examples = payload["examples"]
target_answer = payload["target_answer"]
student_id = payload["student_id"]

# Evaluate
evaluation = evaluate_response(question, sample_answer, examples, target_answer)

# Save
save_result(student_id, evaluation)

print("Evaluation complete and stored.")
