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

def iter_math(split = "train"):
    # The dataset name on Hugging Face is 'hendrycks/math'
    ds = load_dataset("hendrycks/math", "main", split = split)
    
    for i, ex in enumerate(ds):
        q = ex["problem"]
        solution = ex["solution"]
        gold = ex["answer"]

        yield dict(uid = f"math/{split}/{i}",
                   task_type = "math",
                   question = q,
                   solution = solution,
                   gold_answer = gold)