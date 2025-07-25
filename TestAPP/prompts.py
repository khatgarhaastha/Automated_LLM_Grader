# prompts.py
def create_prompt(question, sample_answer, examples, target_answer):
    # Return plain text prompt, not JSON
    return f"""
Evaluate a student's answer based on the question and examples.

Question:
{question}

Sample Answer:
{sample_answer}

Student Response Examples:
Excellent: {examples['excellent']}
Average: {examples['average']}
Poor: {examples['poor']}

Now evaluate this student's answer:
{target_answer}

Provide a brief evaluation and rate it from 1 (Poor) to 5 (Excellent).
"""