import os
import argparse
from src.GeneratorTeacherModel import GeneratorTeacherModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type = str, default = "meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--mode", type = str, default = "boring")    
    ap.add_argument("--dataset", type = str, default = "gsm8k")
    # ap.add_argument("--dataset_mode", type = str, default = "train")
    ap.add_argument("--dataset_mode", type = str, default = "test")
    ap.add_argument("--load_in_8bit", type = bool, default = True)
    # ap.add_argument("--batch_size", type=int, default = 2)
    ap.add_argument("--batch_size", type=int, default = 18)
    ap.add_argument("--max_new_tokens", type = int, default = 256)
    # ap.add_argument("--temperature", type = float, default = 1.2)
    ap.add_argument("--temperature", type = float, default = 0.7)
    ap.add_argument("--top_p", type = float, default = 0.9)
    ap.add_argument("--n", type = int, default = 3)
    ap.add_argument("--repetition_penalty", type = float, default = 1.0)
    args = ap.parse_args()
    args.out_path = f"../datasets/teacher-{args.model.split("/")[0]}/{args.dataset}-{args.mode}-{args.dataset_mode}"
    os.makedirs(f"../datasets/teacher-{args.model.split("/")[1]}/", exist_ok = True)

    ## load dataset to wrap 
    if args.dataset == "gsm8k":
        from src.adapters import iter_gsm8k
        dataset = list(iter_gsm8k(split = args.dataset_mode))
    elif args.dataset == "math":
        from src.adapters import iter_math
        dataset = list(iter_math(split = args.dataset_mode))
    
    ## get teacher model
    teacher = GeneratorTeacherModel(model_name = args.model,
                            dataset_name = args.dataset,
                            mode = args.mode,
                            load_in_8bit = args.load_in_8bit,
                            max_new_tokens = args.max_new_tokens,
                            temperature = args.temperature,
                            top_p = args.top_p,
                            n = args.n,
                            repetition_penalty = args.repetition_penalty)
    
    if args.mode == 'evaluate':
        teacher.evaluate(dataset = dataset, batch_size = args.batch_size)

    else:
        teacher.synthesize(dataset = dataset,
                        out_path = args.out_path,
                        batch_size = args.batch_size)
    
if __name__ == "__main__":
    main()