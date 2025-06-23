import json
from datetime import datetime

def save_result(student_id, evaluation):
    result = {
        "student_id": student_id,
        "timestamp": datetime.utcnow().isoformat(),
        "evaluation": evaluation
    }
    with open(f"results/{student_id}_eval.json", "w") as f:
        json.dump(result, f, indent=4)
