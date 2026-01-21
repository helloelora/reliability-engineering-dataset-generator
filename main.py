import os
import json
import re
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import Dict, List, Optional

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
INPUT_FILE = "Textbook exercices/Reliability.md"
OUTPUT_FILE = "dataset_reliability_verified.jsonl"
REJECTED_FILE = "dataset_rejected_debug.jsonl"

# Parameters
MAX_WORKERS = 10  # Adjust based on your rate limit
MODEL_REASON = "deepseek/deepseek-r1-distill-llama-70b"  # Model optimized for math and reasoning tasks

# Client API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Thread-safe locks for writing and statistics
stats_lock = threading.Lock()
file_lock = threading.Lock()

global_stats = {
    "processed": 0,
    "rejected_syntax": 0,
    "rejected_discrepancy": 0,
    "cost": 0.0
}


REASONING_PROMPT = """
You are a Reliability Engineering Professor. 
You are provided with a Question and its official Target Answer from a textbook.

**Task**:
1. Solve the problem step-by-step yourself (derive the math).
2. **SAFETY CHECK**: Compare your derived result with the provided "Target Answer".
   - If your result matches the Target Answer (within approx 5% margin): Output the reasoning steps.
   - If your result CONTRADICTS the Target Answer: Output "DISCREPANCY_FOUND" in the status field.

**Input Data**:
- Question: {question}
- Target Answer: {answer}

**Output Format**:
Return a valid JSON object ONLY. Do not include the question or answer in the output, only the reasoning and status.
{{
    "status": "MATCH" or "DISCREPANCY_FOUND",
    "reasoning": "We start by identifying the distribution... applying the formula... substitution gives...Therefore, the final answer is:..."
}}
"""


def remove_captions_and_noise(text: str) -> str:
    """
    Cleans text from book structure artifacts:
    1. Figure and Table captions
    2. Recurring headers/footers (e.g., "16 *Applied Reliability*")
    3. Isolated page numbers
    """
    header_regex = r'(?:\d+\s+)?(?:[*_]+)?Applied Reliability(?:[*_]+)?(?:\s+\d+)?'
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped: continue  # Skip empty lines  # Skip empty lines 
        
        if re.fullmatch(header_regex, line_stripped, re.IGNORECASE):
            continue

        if re.match(r'^(?:\*\*)?\s*(?:FIGURE|TABLE)\s+\d+', line_stripped, re.IGNORECASE):
            continue

        if re.match(r'^\d+$', line_stripped):
            continue
            
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines).strip()
    # Remove pattern from the very end of text ($)
    cleaned_text = re.sub(header_regex + r'$', '', cleaned_text, flags=re.IGNORECASE).strip()
    
    return cleaned_text


def has_cross_reference(text: str) -> bool:
    """
    Detects if the question references missing external context.
    Should be called AFTER `remove_captions_and_noise`.
    """
    # Patterns indicating external dependency
    refs = [
        r'example\s+\d',           # "Example 5.3"
        r'exercise\s+\d',          # "Exercise 11.4"
        r'problem\s+\d',           # "Problem 2"
        r'section\s+\d',           # "Section 4.2"
        r'chapter\s+\d',           # "Chapter 3"
        r'appendix',               # "Appendix A"
        r'previous\s+(?:problem|exercise|example)', # "previous example"
        r'data\s+in',              # "data in Example..."
        r'refer\s+to',             # "Refer to..."
        r'based\s+on',             # "Based on..."
        r'shown\s+in',             # "Shown in..."
        r'from\s+(?:example|exercise|problem)',    # "From Example..."
    ]
    
    text_lower = text.lower()
    
    # Special handling: TABLES
    # If the word "table" is present, check if a Markdown table (|---|) exists.
    has_table_word = "table" in text_lower
    has_md_table = bool(re.search(r'\|[\s-]*:?[\s-]{3,}:?[\s-]*\|', text))
    
    # If "table" is mentioned but no visual table exists -> Reject
    if has_table_word and not has_md_table:
        return True 

    # Special handling: FIGURES
    # If "figure" remains after caption cleaning, it's a reference in the text -> Reject
    if "figure" in text_lower:
        return True

    # Check remaining patterns
    for pattern in refs:
        if re.search(pattern, text_lower):
            return True
            
    return False


def clean_spaces(text: str) -> str:
    """Final cleanup for JSON (removes double spaces and line breaks)"""
    if not text: return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def parse_textbook(file_path: str):
    """Extracts questions and answers from the Markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return [], {}

    # --- 1. Extract Answers (FIXED FOR MULTI-LINE) ---
    answers_map = {}
    if "Answers to Selected Exercises" in content:
        _, answers_section = content.split("Answers to Selected Exercises", 1)
        
        lines = answers_section.split('\n')
        
        current_id = None
        current_text = []
        
        new_answer_pattern = re.compile(r'^\s*[-*]?\s*(\d+\.\d+)\.?\s+(.*)')
        
        for line in lines:
            line = line.strip()
            if not line: continue  # Ignore empty lines
            
            match = new_answer_pattern.match(line)
            
            if match:
                if current_id:
                    answers_map[current_id] = " ".join(current_text).strip()
                
                current_id = match.group(1)
                current_text = [match.group(2).strip()]
                
            else:
                if current_id:
                    current_text.append(line)
        
        # Don't forget to save the very last answer from the loop
        if current_id:
            answers_map[current_id] = " ".join(current_text).strip()
            
    else:
        print("Warning: Section 'Answers to Selected Exercises' not found.")
    
    raw_blocks = re.split(r'####\s+\*\*EXERCISE', content)

    candidates = []
    for block in raw_blocks[1:]:
        match_id = re.match(r'\s+(\d+\.\d+)\*\*(.*)', block, re.DOTALL)
        if match_id:
            ex_id = match_id.group(1)
            raw_text = match_id.group(2).strip()
            raw_text = raw_text.split('####')[0].strip()
            raw_text = raw_text.split('### ')[0].strip()
            clean_q_text = remove_captions_and_noise(raw_text) # Votre fonction de nettoyage
            if clean_q_text:
                candidates.append({"id": ex_id, "question": clean_q_text})
            
    return candidates, answers_map

def process_item(item, answer_text):
    q_id = item["id"]
    q_text = item["question"]  # Already cleaned of captions
    a_text = answer_text
    
    # Clean up spacing for LLM submission
    q_clean_spaces = clean_spaces(q_text)
    a_clean_spaces = clean_spaces(a_text)
    
    title = f"Problem {q_id}, Reliability Textbook"

    # 1. Syntax filter (Free): Cross-references
    if has_cross_reference(q_text):
        return {
            "status": "rejected_syntax", 
            "reason": "External reference detected (Example, Figure, Table...)",
            "q_preview": q_clean_spaces[:100]
        }

    # 2. LLM call (Paid): Generation + Verification
    try:
        response = client.chat.completions.create(
            model=MODEL_REASON,
            messages=[{"role": "user", "content": REASONING_PROMPT.format(question=q_clean_spaces, answer=a_clean_spaces)}],
            response_format={"type": "json_object"},
            temperature=0.2  # Low temperature for mathematical rigor
        )
        
        # Track cost
        with stats_lock:
            u = response.usage
            # Rough estimation (DeepSeek pricing varies, adjust according to your provider)
            cost = (u.prompt_tokens/1e6 * 0.35) + (u.completion_tokens/1e6 * 1.40)
            global_stats["cost"] += cost

        # Parse response
        content = response.choices[0].message.content
        result_json = json.loads(content)
        
        status = result_json.get("status", "MATCH")
        reasoning = result_json.get("reasoning", "")

        # 3. Verify LLM verdict
        if status == "DISCREPANCY_FOUND" or "DISCREPANCY_FOUND" in reasoning:
             return {
                 "status": "rejected_discrepancy", 
                 "reason": "LLM found discrepancy with target answer", 
                 "llm_output": reasoning,
                 "id": q_id
             }
        
        # Validate minimum length
        if len(reasoning) < 20:
             return {"status": "rejected_discrepancy", "reason": "Reasoning too short/empty", "id": q_id}

        # SUCCESS: Return the final object
        # Note: We return the original text (clean_spaces) to maintain integrity
        return {
            "status": "success",
            "data": {
                "title": title,
                "question": q_clean_spaces,
                "reasoning": reasoning,
                "answer": a_clean_spaces
            }
        }

    except Exception as e:
        return {"status": "error", "reason": str(e), "id": q_id}


def main():
    # Initial cleanup of output files
    if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)
    if os.path.exists(REJECTED_FILE): os.remove(REJECTED_FILE)

    print(">>> 1. Parsing the Textbook...")
    questions, answers_map = parse_textbook(INPUT_FILE)
    
    # Create valid pairs (Exercises that have both a question and an answer)
    pairs = []
    for q in questions:
        if q["id"] in answers_map:
            pairs.append((q, answers_map[q["id"]]))
    
    print(f">>> {len(pairs)} candidate exercises found (with answers).")
    print(">>> 2. Starting processing...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks
        futures = {executor.submit(process_item, p[0], p[1]): p[0]["id"] for p in pairs}

        for future in as_completed(futures):
            q_id = futures[future]
            try:
                res = future.result()
                
                # --- SUCCESS CASE ---
                if res["status"] == "success":
                    with file_lock:
                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                            f.write(json.dumps(res["data"]) + "\n")
                        global_stats["processed"] += 1
                    print(f"[OK] {q_id}")

                # --- REJECTION CASE (Syntax or Discrepancy) ---
                elif "rejected" in res["status"]:
                    with file_lock:
                        with open(REJECTED_FILE, "a", encoding="utf-8") as f:
                            # Detailed log for debugging
                            log_entry = {
                                "id": q_id, 
                                "status": res["status"],
                                "reason": res.get("reason"),
                                "llm_output": res.get("llm_output", "N/A")
                            }
                            f.write(json.dumps(log_entry) + "\n")
                        
                        if res["status"] == "rejected_syntax": 
                            global_stats["rejected_syntax"] += 1
                        else: 
                            global_stats["rejected_discrepancy"] += 1
                    
                    # Show fewer details for Syntax (very common) than for Discrepancy
                    if res["status"] == "rejected_syntax":
                        print(f"[SKIP-SYNTAX] {q_id}")
                    else:
                        print(f"[SKIP-MATH] {q_id} : Discrepancy found")

                # --- API ERROR CASE ---
                elif res["status"] == "error":
                    print(f"[ERROR] {q_id} : {res.get('reason')}")

            except Exception as e:
                print(f"[CRASH] {q_id} : {e}")

    print("\n" + "="*40)
    print("PROCESSING COMPLETED")
    print("="*40)
    print(f"Final Dataset   : {global_stats['processed']} items (Saved in {OUTPUT_FILE})")
    print(f"Syntax Rejection: {global_stats['rejected_syntax']} items (Ghost questions, refs missing)")
    print(f"Math/LLM Rejection: {global_stats['rejected_discrepancy']} items (Wrong answers, contradictions)")
    print(f"Estimated Cost  : ${global_stats['cost']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()