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

    num_m = NUM_TOKEN_RE.search(tail)
    return num_m.group(0).replace(",", "") if num_m else tail 

def to_template(example):

    q = normalize_text(example["question"])
    sol = normalize_text(example["answer"])
    n = extract_final_answer(sol)

    lines = [ln for ln in sol.split("\n")]
    if lines and lines[-1].lstrip().startswith("####"):
        rationale = "\n".join(lines[:-1]).strip()
    else:
        rationale = re.sub(r"####\s*"+re.escape(n)+r"\s*$", "", sol, flags=re.S).strip()

    return {"question": q,
        "assistant_target": f"{rationale}\n<answer>{n}</answer>",
        "golden_answer": n}
    
def to_template_trippy(example): 
    
    q = normalize_text(example["question"])
    sol = normalize_text(example["assistant_answer"])
    n = normalize_text(example["gold_answer"])

    return {"question": q,
        "assistant_target": sol,
        "golden_answer": n}