# adapters/gsm8k.py
from datasets import load_dataset, concatenate_datasets
import re

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

def iter_math(split="train"):

    subjects = ['algebra', 'counting_and_probability', 'geometry', 
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    datasets = []
    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split = split)
        datasets.append(ds)
    combined_ds = concatenate_datasets(datasets)

    for i, ex in enumerate(ds):
        q = ex["problem"]
        solution = ex["solution"]
        boxed_match = re.search(r'\\boxed\{([^}]*)\}', solution)
        gold = boxed_match.group(1) if boxed_match else None
        
        yield dict(
            uid = f"math/{split}/{i}",
            task_type = "math",
            question = q,
            solution = solution,
            gold_answer = gold)