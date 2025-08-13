import argparse
from src.synthetic_data.adapters.gsm8k import iter_gsm8k
from src.synthetic_data.DataGenerator import DataGenerator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", 
                    type = str, 
                    default = "meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--mode", 
                    type = str, 
                    choices = ["single", "iter"], default = "single")
    ap.add_argument("--limit", type = int, default = 8000)
    ap.add_argument("--dataset", type = str, default = "gsm8k")
    ap.add_argument("--load_in_8bit", type = bool, default = False)
    ap.add_argument("--batch_size", type=int, default = 24)
    ap.add_argument("--max_new_tokens", type = int, default = 256)
    ap.add_argument("--temperature", type = float, default = 0.9)
    ap.add_argument("--top_p", type = float, default = 0.9)
    ap.add_argument("--n", type = int, default = 1)
    ap.add_argument("--stop_sequences", type = str, nargs = "*", default = None) 
    ap.add_argument("--repetition_penalty", type = float, default = 1.0)
    ap.add_argument("--torch_dtype", type = str, default = "auto")
    args = ap.parse_args()
    args.out_path = f"../datasets/{args.dataset}-test"

    dataset = list(iter_gsm8k(split = "train"))
    
    teacher = DataGenerator(
            model_name = args.model,
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