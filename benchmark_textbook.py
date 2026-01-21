import json
import os
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it before running this script.")
INPUT_FILE = "dataset_reliability_verified.jsonl"
OUTPUT_FILE = "benchmark_results_open_ended.json"

# Judge model and candidate models to test
JUDGE_MODEL = "openai/gpt-4o-mini"
MODELS_TO_TEST = [
    "qwen/qwen3-14b",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-nemo"
]

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 2  # Seconds

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def load_jsonl(filepath):
    """Load questions from the JSONL file."""
    questions = []
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    questions.append(json.loads(line))
                except: pass
    return questions

def ask_candidate_model(model, question_data):
    """Ask the candidate model to solve the problem using exponential backoff strategy."""
    prompt = f"""You are a Reliability Engineering expert.
Solve the following problem. 

Question: {question_data['question']}

Provide a clear, step-by-step reasoning.
IMPORTANT: You must state your final answer clearly at the very end, starting with "Final Answer:".
"""
    
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
                timeout=90 
            )
            
            # Optional check for completion reason
            finish_reason = completion.choices[0].finish_reason
            content = completion.choices[0].message.content
            
            if finish_reason == "length":
                print(f"Warning: [{model}] Response truncated by max_tokens limit.")
                # Could decide to return what we have or treat as an error
            
            return content

        except Exception as e:
            wait_time = BASE_DELAY * (2 ** attempt)
            print(f"Warning: [Candidate Error {model}] Attempt {attempt+1}/{MAX_RETRIES}. Error: {e}")
            time.sleep(wait_time)
            
    return "ERR_API"

def evaluate_answer(question, target_answer, candidate_answer):
    """
    Evaluate the candidate answer using exponential backoff strategy.
    """
    if candidate_answer == "ERR_API":
        return False, "API Error (Candidate failed to answer)"

    judge_prompt = f"""You are an impartial exam grader for Reliability Engineering.

**Task**: Compare the Student's Answer with the Target Answer (Ground Truth).

**Context**:
- Question: {question}
- Target Answer (Correct): {target_answer}
- Student's Answer: {candidate_answer}

**Grading Rules**:
1. **Mathematics**: If the student's result is numerically close (within ~5% margin), mark as CORRECT.
2. **Equivalence**: If the student derives a formula that is mathematically equivalent to the target (e.g., "1 - exp(-lt)" vs "1 - e^(-lambda*t)"), mark as CORRECT.
3. **Reasoning**: Ignore minor wording differences. Focus on the final result/conclusion.

**Output Format**:
Reply with a SINGLE JSON OBJECT:
{{
  "is_correct": boolean,
  "explanation": "Short reason why"
}}
"""
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=30
            )
            content = completion.choices[0].message.content
            result = json.loads(content)
            return result.get("is_correct", False), result.get("explanation", "No explanation")
        except Exception as e:
            wait_time = BASE_DELAY * (2 ** attempt)
            print(f"Warning: [Judge Error] Attempt {attempt+1}/{MAX_RETRIES}. Error: {e}")
            print(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            
    return False, "Judge API Error"

def process_single_item(model_name, q):
    """Orchestrate: Ask the question, receive response, and have judge evaluate it."""
    q_title = q.get('title', 'Unknown Title')
    target_answer = q.get('answer')
    
    # 1. Get candidate's response
    candidate_resp = ask_candidate_model(model_name, q)
    
    # 2. Have judge evaluate
    is_correct, explanation = evaluate_answer(q['question'], target_answer, candidate_resp)
    
    return {
        "question_title": q_title,
        "target_answer": target_answer,
        "model_prediction": candidate_resp,
        "is_correct": is_correct,
        "judge_explanation": explanation
    }

def run_benchmark():
    # Load questions
    questions = load_jsonl(INPUT_FILE)
    if not questions: return

    # Load existing results
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n===========================================")
        print(f"Evaluating model: {model_name}")
        print(f"==========================================")

        if model_name not in results:
            results[model_name] = {"score": 0, "total": 0, "details": []}
        
        # Filter questions that are already done (resume on error)
        existing_titles = {d['question_title'] for d in results[model_name]['details']}
        todo = [q for q in questions if q.get('title') not in existing_titles]
        
        print(f"Processing {len(todo)} questions...")

        # Note: With 5 workers and long retries, the script may take time in case of API issues.
        # This is normal and intentional for robustness.
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_q = {executor.submit(process_single_item, model_name, q): q for q in todo}
            
            count = 0
            for future in as_completed(future_to_q):
                try:
                    res = future.result()
                    
                    status = "[OK]" if res['is_correct'] else "[FAIL]"
                    # If persistent API error after 5 attempts
                    if res['model_prediction'] == "ERR_API": status = "[ERROR]"
                        
                    print(f"{status} {res['question_title']} | Judge: {res['judge_explanation']}")
                    
                    results[model_name]['details'].append(res)
                    count += 1
                    
                    if count % 5 == 0:
                        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)

                except Exception as e:
                    print(f"Warning: Critical error on a question: {e}")

        # Recalculate final score
        details = results[model_name]['details']
        # API errors are counted as incorrect (or you could exclude them)
        # Currently counting as incorrect if API error
        correct = sum(1 for d in details if d.get('is_correct'))
        total = len(details)
        if total > 0:
            print(f"Final Score {model_name}: {correct}/{total} ({correct/total*100:.2f}%)")
        
        # Save final results for this model
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_benchmark()