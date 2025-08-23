import os
import argparse
from datasets import load_dataset

from src.TrainerTrippyModel import TrainerTrippyModel
from src.adapters import iter_gsm8k, iter_trippy
from src.chat_utils import to_template, to_template_trippy

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def parse_args():
    ap = argparse.ArgumentParser()

    ## general
    ap.add_argument("--note", type = str, default = "new-test")
    ap.add_argument("--finetuning", action='store_true')
    ap.add_argument("--testing", action='store_false')  
    ap.add_argument("--inference", action='store_false')

    ## models
    ap.add_argument("--model_name", type = str,
                    default = "mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--teacher_name", type = str,
                    default = "meta-llama/Meta-Llama-3-3B-Instruct")
    ## dataset
    ap.add_argument("--dataset_name", type = str,
                    default = "gsm8k")
    ap.add_argument("--mode", type = str, help = "trippy") ## the type of dataset
    
    ## prompting mode
    ap.add_argument("--SPECIAL_TOKENS", type = list, default = [])
    ap.add_argument("--seq_length", type = int, default = 1536)

    ## fine-tuning
    ap.add_argument("--epochs", type = int, default = 3)
    ap.add_argument("--lr", type = float, default = 2e-4)
    ap.add_argument("--per_device_bs", type = int, default = 1)
    ap.add_argument("--grad_accum", type = int, default = 32)
    ap.add_argument("--lora_r", type = int, default = 16)
    ap.add_argument("--lora_alpha", type = int, default = 32)
    ap.add_argument("--lora_dropout", type = float, default = 0.05)
    ap.add_argument("--save_steps", type = int, default = 1000)
    ap.add_argument("--logging_steps", type = int, default = 20)

    # generation / inference
    ap.add_argument("--load_adapter", action='store_false')
    ap.add_argument("--temperature", type=float, default = 0.7)
    ap.add_argument("--top_p", type=float, default = 0.9)
    ap.add_argument("--max_new_tokens", type=int, default = 256)
    ap.add_argument("--batch_size", type=int, default = 1)
    ap.add_argument("--n", type=int, default = 3)

    return ap.parse_args()

def main():
    args = parse_args()
    args.output_dir = f"../runs-sft/model-{args.model_name.split("/")[1]}/teacher-{args.teacher_name.split("/")[1]}/{args.dataset_name}-{args.mode}-{args.note}"
    os.makedirs(args.output_dir, exist_ok = True)

    ## load dataset
    logger.info(f"loading dataset {args.dataset_name}")

    if args.mode == 'default':
        ds = load_dataset("gsm8k", "main")
        split = ds["train"].train_test_split(test_size = 0.1, seed = 42)
        train_set, val_set = split["train"], split["test"]
        train_set = train_set.map(to_template, remove_columns = train_set.column_names)
        val_set = val_set.map(to_template, remove_columns = val_set.column_names)
        test_set = list(iter_gsm8k(split = "test"))

    else:
        train_file = f"../datasets/teacher-{args.teacher_name.split('/')[1]}/{args.dataset_name}-{args.mode}-train.json"
        test_file  = f"../datasets/teacher-{args.teacher_name.split('/')[1]}/{args.dataset_name}-{args.mode}-test.json"
        raw_train = load_dataset("json", data_files = {"train": train_file})["train"]
        split = raw_train.train_test_split(test_size = 0.1, seed = 42)
        train_set_raw, val_set_raw = split["train"], split["test"]
        raw_test  = load_dataset("json", data_files = {"test": test_file})["test"]
        train_set = train_set_raw.map(to_template_trippy, remove_columns = train_set_raw.column_names)
        val_set   = val_set_raw.map(to_template_trippy, remove_columns = val_set_raw.column_names)
        test_set  = raw_test.map(to_template_trippy, remove_columns = raw_test.column_names)
        logger.info(f"DEBUG: train set example: \n {train_set[0]}")
        logger.info(f"DEBUG: val set example: \n {val_set[0]}")
        logger.info(f"DEBUG: test set example: \n {test_set[0]}")

    ## get trainer with model and tokenizer
    trainer = TrainerTrippyModel(
            finetuning = args.finetuning,
            testing = args.testing,
            inference = args.inference,
            model_name = args.model_name,
            dataset_name = args.dataset_name,
            mode = args.mode,
            output_dir = args.output_dir,
            SPECIAL_TOKENS = args.SPECIAL_TOKENS,
            seq_length = args.seq_length,
            epochs = args.epochs,
            lr = args.lr,
            per_device_bs = args.per_device_bs,
            grad_accum = args.grad_accum,
            lora_r = args.lora_r,
            lora_alpha = args.lora_alpha,
            lora_dropout = args.lora_dropout,
            save_steps = args.save_steps,
            logging_steps = args.logging_steps,
            load_adapter = args.load_adapter,     
            do_sample = True,
            temperature= args.temperature,
            top_p = args.top_p,
            max_new_tokens = args.max_new_tokens,
            n = args.n)
    
    ## fine tuning
    if args.finetuning:
        trainer.run(train_set = train_set, validation_set = val_set)

    ## testing
    if args.testing:
        trainer.test(dataset = test_set, batch_size = args.batch_size)
    
if __name__ == "__main__":
    main()