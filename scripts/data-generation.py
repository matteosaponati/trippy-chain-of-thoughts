import os
import argparse
from datasets import load_dataset

from src.GeneratorModel import GeneratorModel
from src.chat_utils import to_template

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type = str, default = "Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mode", type = str, default = "boring")    
    ap.add_argument("--dataset", type = str, default = "gsm8k")
    ap.add_argument("--split", type = str, default = "train")
    ap.add_argument("--load_in_4bit", type = bool, default = True)
    ap.add_argument("--batch_size", type=int, default = 10)
    ap.add_argument("--max_new_tokens", type = int, default = 256)
    ap.add_argument("--seq_length", type = int, default = 1024)
    ap.add_argument("--temperature", type = float, default = 0.9)
    ap.add_argument("--top_p", type = float, default = 0.9)
    args = ap.parse_args()
    args.output_dir = f"../datasets/teacher-{args.model.split("/")[1]}"
    os.makedirs(args.output_dir, exist_ok = True)

    ## load dataset to wrap 
    ds = load_dataset("gsm8k", "main")[args.split]
    ds = ds.map(to_template, remove_columns = ds.column_names)
    
    ## get teacher model
    teacher = GeneratorModel(model_name = args.model,
                            dataset_name = args.dataset,
                            mode = args.mode,
                            output_dir = args.output_dir,
                            split = args.split,
                            load_in_4bit = args.load_in_4bit,
                            max_new_tokens = args.max_new_tokens,
                            seq_length = args.seq_length,
                            temperature = args.temperature,
                            top_p = args.top_p)
    ## generate wrapped dataset
    teacher.generate(dataset = ds, batch_size = args.batch_size)
    
if __name__ == "__main__":
    main()