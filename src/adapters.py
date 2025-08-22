# adapters/gsm8k.py
from datasets import load_dataset

def iter_gsm8k(split = "train"):
    ds = load_dataset("gsm8k", "main", split = split)
    
    for i, ex in enumerate(ds):
        q = ex["question"]
        
        if "####" in ex["answer"]:
            gold = ex["answer"].split("####")[-1].strip()
        else:
            gold = ex["answer"].strip()

        yield dict(uid = f"gsm8k/{split}/{i}",
                   task_type = "math",
                   question = q,
                   gold_answer = gold)