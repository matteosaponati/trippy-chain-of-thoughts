from typing import List, Dict, Optional

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
                    {"role":"user", "content": USER_TEMPLATE.format(question = question, answer = answer)}]

def to_chat(messages: List[Dict],
            tokenizer) -> str:
    """Take a list of dictionary (each representing a prompt in chat format)
    and convert it into a single text for the model to understand the conversation."""
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

def to_wrapped_data(results: List[str], 
                    current_batch: Dict, 
                    dataset_name: str):
    """Take the best trip_before, answer, end for a given answer and wrap the new dataset."""                    
    return [{"messages": [{"role": "user", "content": f"<problem>\n{ex['question']}\n</problem>"},
                    {"role": "assistant", "content": (f"<trip_before>{r[0]}</trip_before>\n"f"<answer>{r[1]}</answer>\n"f"<end>{r[2]}</end>")}],
                    "meta": {"source": dataset_name, "id": ex["uid"], "task_type": ex["task_type"]}}
                    for r, ex in zip(results, current_batch) if r is not None]
    