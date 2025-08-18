import argparse
from src.synthetic_data.DataGenerator import DataGenerator
from src.synthetic_data.adapters import iter_gsm8k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", 
                    type = str, 
                    default = "meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--mode", 
                    type = str, 
                    default = "boring")    
    ap.add_argument("--limit", type = int, default = 8000)
    ap.add_argument("--dataset", type = str, default = "gsm8k")
    ap.add_argument("--dataset_mode", type = str, default = "test")
    ap.add_argument("--load_in_8bit", type = bool, default = False)
    ap.add_argument("--batch_size", type=int, default = 18)
    ap.add_argument("--max_new_tokens", type = int, default = 256)
    ap.add_argument("--temperature", type = float, default = 0.9)
    ap.add_argument("--top_p", type = float, default = 0.9)
    ap.add_argument("--n", type = int, default = 1)
    ap.add_argument("--stop_sequences", type = str, nargs = "*", default = None) 
    ap.add_argument("--repetition_penalty", type = float, default = 1.0)
    ap.add_argument("--torch_dtype", type = str, default = "auto")
    args = ap.parse_args()
    args.out_path = f"../datasets/{args.dataset}-{args.mode}-{args.dataset_mode}"

    ## load dataset to wrap 
    if args.dataset == "gsm8k":
        from src.synthetic_data.adapters import iter_gsm8k
        dataset = list(iter_gsm8k(split = args.dataset_mode))
    elif args.dataset == "math":
        from src.synthetic_data.adapters import iter_math
        dataset = list(iter_math(split = args.dataset_mode))
    
    ## create data generator
    teacher = DataGenerator(
            model_name = args.model,
            dataset_name = args.dataset,
            mode = args.mode,
            torch_dtype = args.torch_dtype,
            load_in_8bit = args.load_in_8bit,
            max_new_tokens = args.max_new_tokens,
            temperature = args.temperature,
            top_p = args.top_p,
            n = args.n,
            stop_sequences = args.stop_sequences,
            repetition_penalty = args.repetition_penalty)

    teacher.synthesize(dataset = dataset,
                       out_path = args.out_path,
                       batch_size = args.batch_size)
    
if __name__ == "__main__":
    main()