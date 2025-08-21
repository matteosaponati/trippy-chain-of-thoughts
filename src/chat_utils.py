from typing import List, Dict
import re

LAST_HASH_RE = re.compile(r"####\s*(.+?)\s*$", re.S)   # multiline/end-tolerant
NUM_TOKEN_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+/\d+")

def normalize_text(s: str) -> str:
    """normalize weird spaces and line endings"""
    return (s.replace("\r\n", "\n")
         .replace("\r", "\n")
         .replace("\u00A0", " ") 
         .replace("\u2009", " ") 
         .strip())

def extract_final_answer(sol: str) -> str:
    sol_norm = normalize_text(sol)
    m = LAST_HASH_RE.search(sol_norm)
    if not m:
        if "####" in sol_norm:
            tail = sol_norm.split("####")[-1].strip()
        else:
            raise ValueError("No '####' marker found.")
    else:
        tail = m.group(1).strip()

    # Optionally sanitize the numeric (keep original tail if you prefer)
    num_m = NUM_TOKEN_RE.search(tail)
    return num_m.group(0).replace(",", "") if num_m else tail  # strip commas if present

def to_template(example):
    q = normalize_text(example["question"])
    sol = normalize_text(example["answer"])

    try:
        n = extract_final_answer(sol)  # "10", "3.5", "1/2", etc.
    except Exception as e:
        # Propagate enough info to debug the offending row
        raise AssertionError(f"No final #### number found.\nQUESTION:\n{q}\nANSWER_RAW:\n{sol}\nERROR:{e}")

    # Remove the last '#### ...' line from the rationale body if present
    # Safer approach: splitlines and drop the line that STARTS with '####'
    lines = [ln for ln in sol.split("\n")]
    if lines and lines[-1].lstrip().startswith("####"):
        rationale = "\n".join(lines[:-1]).strip()
    else:
        # If it's embedded, just remove the substring '#### <tail>' once from the end
        rationale = re.sub(r"####\s*"+re.escape(n)+r"\s*$", "", sol, flags=re.S).strip()

    assistant_target = f"{rationale}\n<answer>{n}</answer>"

    return {
        "question": q,
        "assistant_target": assistant_target,
        "final_answer": n,
    }
def to_messages(question: str, 
                answer: str,
                SYSTEM_PROMPT: str,
                USER_TEMPLATE: str,
                mode: str) -> List[Dict]:
    """Convert a question string into a list of messages in chat format."""
    if mode == 'evaluate':
            return [{"role":"system", "content": SYSTEM_PROMPT},
                    {"role":"user", "content": USER_TEMPLATE.format(question = question)}]
    else:
            return [{"role":"system", "content": SYSTEM_PROMPT},
                    {"role":"user", "content": USER_TEMPLATE.format(question = question, answer = answer)}] # this is not ok for trippy fine-tuning

def to_chat(messages: List[Dict],
            tokenizer, add_generation_prompt: bool = True) -> str:
    """Take a list of dictionary (each representing a prompt in chat format)
    and convert it into a single text for the model to understand the conversation."""
    return tokenizer.apply_chat_template(messages, 
                                         tokenize = False, 
                                         add_generation_prompt = add_generation_prompt)

def to_wrapped_data(results: List[str], 
                    current_batch: Dict, 
                    dataset_name: str):
    """Take the best trip_before, answer, end for a given answer and wrap the new dataset."""                    
    return [{"messages": [{"role": "user", "content": f"<problem>\n{ex['question']}\n</problem>"},
                    {"role": "assistant", "content": (f"<trip_before>{r[0]}</trip_before>\n"f"<answer>{r[1]}</answer>\n"f"<end>{r[2]}</end>")}],
                    "meta": {"source": dataset_name, "id": ex["uid"], "task_type": ex["task_type"]}}
                    for r, ex in zip(results, current_batch) if r is not None]
    