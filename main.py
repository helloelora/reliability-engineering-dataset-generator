"""
Reliability Engineering dataset generator

A three-stage pipeline for extracting, augmenting and solving 
reliability engineering exercises from OCR-processed textbooks.

Pipeline stages:
1. Extract: Identify exercises from textbook chunks
2. Augment: Rewrite questions to be self-contained
3. Solve: Generate step-by-step reasoning with verification
"""

import os
import json
import re
import glob
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import List, Dict, Any, Optional


# Configuration
OPENROUTER_API_KEY = ""
OUTPUT_FILE = "dataset_reliability_augmented.jsonl" 
REJECTED_FILE = "dataset_rejected.jsonl"

MAX_WORKERS = 5
TEXTBOOKS_FOLDER = "reliability books ocr/mistral ocr" 

MODEL_EXTRACT = "openai/gpt-4o-mini"            
MODEL_AUGMENT = "openai/gpt-4o-mini"            
MODEL_REASON = "deepseek/deepseek-r1-distill-llama-70b" 

global_stats = {
    "total_cost": 0.0,
    "saved_items": 0,
    "rejected_items": 0,
    "api_errors": 0
}
stats_lock = threading.Lock()
file_lock = threading.Lock()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)


EXTRACT_PROMPT = """
You are a strict data extraction specialist for Reliability Engineering textbooks.
RULES:
1. **Identify Exercises**: Look for "Example", "Problem", "Exercise", "Question".
2. **SEPARATE STRICTLY**:
   - `question_clean`: The problem statement ONLY. Remove answers/hints.
   - `final_answer`: The final result found in the text.
   - `provided_reasoning`: The step-by-step solution from the text (OCR). If none, null.
3. **Output JSON**: {"exercises": [{"source_id": "...", "question_clean": "...", "final_answer": "...", "provided_reasoning": "..."}]}
"""

AUGMENT_PROMPT = """
You are a Textbook Editor. Your goal is to rewrite an exercise to make it completely STANDALONE and SELF-CONTAINED.

Input Data:
- **Draft Question**: {q}
- **Context/Snippet**: "{context}"

TASKS:
1. **Check for Missing Data**: If the question relies on a Table, Figure, Chart, or Appendix not explicitly fully described in the text, REJECT IT.
2. **Inject Parameters**: If the question refers to values found in the context (e.g. "Calculate reliability for the system above"), you MUST rewrite the question to include these values explicitly (e.g. "Calculate reliability for a system with lambda=0.01").
3. **Cleanup**: Remove references like "As seen in Example 4.1" or "From the previous section".

OUTPUT JSON ONLY:
{{
   "status": "valid" OR "rejected",
   "augmented_question": "The fully rewritten, standalone question...",
   "rejection_reason": "Only if rejected (e.g. Missing Table)"
}}
"""

SOLVER_PROMPT = """
You are a Reliability Engineering Professor. Solve the following problem step-by-step.

**Problem**: 
{q}

**Target Answer (for verification only)**: 
{a}

INSTRUCTIONS:
1. **Derive the solution** step-by-step using standard LaTeX for math.
2. **Safety Check**: Compare your final result with the Target Answer.
   - If consistent (approx 5% error margin): Output the reasoning.
   - If FUNDAMENTALLY different (contradiction): Output "DISCREPANCY_FOUND".

OUTPUT JSON ONLY:
{{
   "reasoning": "The step-by-step derivation...",
   "final_answer_check": "The result you found"
}}
"""


def retry_api_call(func):
    def wrapper(*args, **kwargs):
        retries = 3
        base_delay = 2
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if "context length" in error_msg or "too large" in error_msg:
                    return None
                if i < retries - 1:
                    time.sleep(base_delay * (2 ** i) + random.uniform(0, 1))
                else:
                    with stats_lock: global_stats["api_errors"] += 1
                    return None
    return wrapper


def is_tautology(question: str, answer: str) -> bool:
    q_lower = question.lower()
    a_clean = str(answer).strip()
    if len(a_clean) > 3 and a_clean in q_lower:
        return True
    return False

def is_context_leak(text: str) -> bool:
    forbidden = [
        "refer to the context", "provided in the context", 
        "as shown in the above", "context snippet"
    ]
    t_lower = text.lower()
    return any(phrase in t_lower for phrase in forbidden)

def is_ghost_question(question: str) -> bool:
    """Check if question references external elements (tables, figures, etc.)"""
    ghost_words = [
        "table", "figure", "chart", "plot", "graph", "shown below", 
        "refer to", "see above", "appendix", "section", "chapter"
    ]
    q_lower = question.lower()
    return any(word in q_lower for word in ghost_words)

def is_discrepancy(reasoning: str) -> bool:
    forbidden = ["discrepancy_found", "reject_missing_data", "does not match the target"]
    r_lower = reasoning.lower()
    return any(phrase in r_lower for phrase in forbidden)


def sanitize_text(text: Any) -> str:
    if text is None: return ""
    if isinstance(text, (dict, list)): return str(text)
    text = str(text)
    text = text.replace('\u0000', ' infinity ') 
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text) 
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fix_latex_artifacts(text: str) -> str:
    if not text: return ""
    replacements = {
        r'\\text{sqrt}': r'\\sqrt', r'\\text{pi}': r'\\pi',
        r'\\text{sigma}': r'\\sigma', r'\\text{mu}': r'\\mu',
        r'\\text{lambda}': r'\\lambda', r'\\text{exp}': r'\\exp', 
        r'\\text{ln}': r'\\ln', r'\\cdot': r' \\cdot '  
    }
    for bad, good in replacements.items():
        try: text = re.sub(bad, good, text, flags=re.IGNORECASE)
        except: pass
    text = re.sub(r'\\text\{([a-zA-Z])\}', r'\1', text)
    return text

def parse_json_response(content: str) -> Dict:
    try:
        clean_content = re.sub(r'^```json\s*', '', content.strip())
        clean_content = re.sub(r'\s*```$', '', clean_content)
        return json.loads(clean_content)
    except:
        return None

def update_cost(model_name, usage):
    if not usage: return
    prices = {
        MODEL_EXTRACT: {"input": 0.15, "output": 0.60},
        MODEL_AUGMENT: {"input": 0.15, "output": 0.60},
        MODEL_REASON:  {"input": 0.35, "output": 1.40} 
    }
    p = prices.get(model_name, {"input": 0, "output": 0})
    cost = (usage.prompt_tokens/1e6 * p["input"]) + (usage.completion_tokens/1e6 * p["output"])
    with stats_lock: global_stats["total_cost"] += cost

def save_entry(entry: Dict, accepted: bool):
    target_file = OUTPUT_FILE if accepted else REJECTED_FILE
    with file_lock:
        try:
            with open(target_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if accepted: global_stats["saved_items"] += 1
            else: global_stats["rejected_items"] += 1
            tot = global_stats["saved_items"] + global_stats["rejected_items"]
            if tot % 5 == 0:
                print(f"Stats: OK {global_stats['saved_items']} | REJECT {global_stats['rejected_items']} | Cost ${global_stats['total_cost']:.4f}")
        except Exception as e:
            print(f"Error saving entry: {e}")


def get_sliding_chunks(text: str, chunk_size: int = 12000, overlap: int = 1500) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            search_zone = text[end - 500 : end] 
            last_newline = search_zone.rfind('\n')
            if last_newline != -1: end = (end - 500) + last_newline
        chunks.append(text[start:end])
        start = end - overlap
        if start >= text_len or end >= text_len: break
    return chunks


@retry_api_call
def call_extraction(chunk):
    return client.chat.completions.create(
        model=MODEL_EXTRACT,
        messages=[{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": chunk}],
        response_format={"type": "json_object"}, timeout=60
    )

@retry_api_call
def call_augmentation(q, context):
    prompt = AUGMENT_PROMPT.format(q=q, context=context)
    return client.chat.completions.create(
        model=MODEL_AUGMENT,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, timeout=60
    )

@retry_api_call
def call_solver(q, a):
    prompt = SOLVER_PROMPT.format(q=q, a=a)
    return client.chat.completions.create(
        model=MODEL_REASON,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=0.3, timeout=180
    )


def process_single_file(file_path: str):
    filename = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f: raw_text = f.read()
        chunks = get_sliding_chunks(raw_text)
    except: return

    raw_exercises = []
    for chunk in chunks:
        completion = call_extraction(chunk)
        if completion:
            update_cost(MODEL_EXTRACT, completion.usage)
            try:
                content = completion.choices[0].message.content
                data = parse_json_response(content) 
                
                exercises_list = []
                
                if isinstance(data, dict):
                    exercises_list = data.get("exercises", [])
                    if not exercises_list and "question_clean" in data:
                        exercises_list = [data]

                elif isinstance(data, list):
                    exercises_list = data
                
                for item in exercises_list:
                    if isinstance(item, dict) and item.get("question_clean") and item.get("final_answer"):
                        raw_exercises.append(item)
                        
            except Exception:
                pass

    unique_exercises = []
    seen = set()
    for ex in raw_exercises:
        clean_q = sanitize_text(ex["question_clean"])
        k = clean_q[:100].replace(" ", "").lower()
        if k not in seen:
            unique_exercises.append(ex)
            seen.add(k)

    if not unique_exercises: return
    print(f"> {filename}: Found {len(unique_exercises)} candidates. Processing...")

    for i, ex in enumerate(unique_exercises):
        q_raw = sanitize_text(ex["question_clean"])
        a = sanitize_text(ex["final_answer"])
        r_ocr = sanitize_text(ex.get("provided_reasoning", ""))
        context_str = r_ocr if len(r_ocr) > 10 else "No context provided."

        if not re.search(r'\d', str(a)) or len(str(a)) > 150: continue

        aug_completion = call_augmentation(q_raw, context_str)
        if not aug_completion: continue
        update_cost(MODEL_AUGMENT, aug_completion.usage)
        
        aug_data = parse_json_response(aug_completion.choices[0].message.content)
        if not aug_data: continue

        if aug_data.get("status") == "rejected":
            save_entry({
                "source_file": filename, "original_question": q_raw,
                "type": "ghost_rejected_step2",
                "fail_reason": aug_data.get("rejection_reason")
            }, False)
            continue
        
        gen_q = fix_latex_artifacts(aug_data.get("augmented_question", ""))
        
        if is_ghost_question(gen_q):
            save_entry({"source_file": filename, "question": gen_q, "fail_reason": "Ghost Question"}, False)
            continue

        solve_completion = call_solver(gen_q, a)
        if not solve_completion: continue
        update_cost(MODEL_REASON, solve_completion.usage)

        solve_data = parse_json_response(solve_completion.choices[0].message.content)
        if not solve_data: continue

        gen_r = fix_latex_artifacts(solve_data.get("reasoning", ""))

        gen_r = gen_r.replace("Based on the provided values, ", "")
        gen_r = gen_r.replace("From the provided context, ", "")
        gen_r = gen_r.replace("In the provided text, ", "")
        gen_r = gen_r.strip()

        accepted = True
        fail_reason = None

        if is_discrepancy(gen_r):
            accepted = False
            fail_reason = "Discrepancy"
        elif is_context_leak(gen_r):
            accepted = False
            fail_reason = "Leak"
        elif is_tautology(gen_q, a):
            accepted = False
            fail_reason = "Tautology"
        elif len(gen_r) < 50 and not any(x in gen_r for x in ['\\', '=', '+', '*', '/', '>', '<', '^']):
            accepted = False
            fail_reason = "Reasoning too short/no math"

        final_entry = {
            "source_file": filename,
            "original_question": q_raw,
            "question": gen_q,
            "reasoning": gen_r,
            "answer": a,
            "type": "synthetic_3step",
            "quality_flag": "ok" if accepted else "rejected",
            "fail_reason": fail_reason
        }
        save_entry(final_entry, accepted)

    print(f"{filename} Finished.")

if __name__ == "__main__":
    files = glob.glob(os.path.join(TEXTBOOKS_FOLDER, "*.md"))
    done_files = set()
    for fpath in [OUTPUT_FILE, REJECTED_FILE]:
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        try: done_files.add(json.loads(line)["source_file"])
                        except: pass
            except: pass
            
    todo = [f for f in files if os.path.basename(f) not in done_files]
    print(f"{len(todo)} files remaining to process.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_file, f): f for f in todo}
        for future in as_completed(futures):
            try: future.result()
            except Exception as e: print(f"Crash: {e}")

    print("finished")
    print(f"Total valid: {global_stats['saved_items']} | Cost: ${global_stats['total_cost']:.4f}")