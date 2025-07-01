import json
import os
from datetime import datetime

def save_result(student_id, evaluation):
    # Ensure the results directory exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    result = {
        "student_id": student_id,
        "timestamp": datetime.utcnow().isoformat(),
        "evaluation": evaluation
    }
    with open(f"results/{student_id}_eval.json", "w") as f:
        json.dump(result, f, indent=4)