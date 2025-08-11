import argparse
from src.synthetic_data.adapters.gsm8k import iter_gsm8k
from src.synthetic_data.DataGenerator import DataGenerator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", 
                    type = str, 
                    default = "meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--mode", 
                    type = str, 
                    choices = ["single", "iter"], default = "single")
    ap.add_argument("--limit", type = int, default = 10000)
    ap.add_argument("--dataset", type = str, default = "gsm8k")
    ap.add_argument("--load_in_4bit", type = bool, default = False)
    ap.add_argument("--max_new_tokens", type = int, default = 384)
    ap.add_argument("--temperature", type = float, default = 0.9)
    ap.add_argument("--top_p", type = float, default = 0.9)
    ap.add_argument("--top_k", type = int, default = None)
    ap.add_argument("--n", type = int, default = 1)
    ap.add_argument("--stop_sequences", type = str, nargs = "*", default = None) 
    ap.add_argument("--repetition_penalty", type = float, default = 1.0)
    args = ap.parse_args()
    args.out_path = f"../datasets/{args.dataset}"

    dataset = list(iter_gsm8k(split = "train"))
    teacher = DataGenerator(
            model_name = args.model,
            device = args.device,
            torch_dtype = args.torch_dtype,
            cache_dir = args.cache_dir,
            load_in_4bit = args.load_in_4bit,
            max_new_tokens = args.max_new_tokens,
            temperature = args.temperature,
            top_p = args.top_p,
            top_k = args.top_k,
            n = args.n,
            stop_sequences = args.stop_sequences,
            repetition_penalty = args.repetition_penalty)
    teacher.synthesize(dataset = dataset,
                       out_path = args.out_path)
    
if __name__ == "__main__":
    main()