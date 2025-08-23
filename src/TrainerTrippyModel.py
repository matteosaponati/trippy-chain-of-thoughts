import torch
import os
import re
import time
import json
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from peft import PeftConfig, PeftModel
import logging

from src.filters import is_correct
from src.ft_utils import config_LoRA, config_SFT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

class TrainerTrippyModel:

    def __init__(self,
            finetuning: bool = True,
            testing: bool = False,
            inference: bool = False,
            model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",   ## the checkpoint used as teaching model
            dataset_name: str = "gsm8k",                               ## the dataset 
            mode: str = 'trippy',
            output_dir: str = "",
            SPECIAL_TOKENS: list = [],
            seq_length: int = 1536,                                    ## the maximum sequence length of the model
            epochs: int = 1,                                           ## number of epochs
            lr: float = 2e-4,
            per_device_bs: int = 1,
            grad_accum: int = 32,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = .1,
            save_steps: int = 1000,
            logging_steps: int = 20,
            load_adapter: bool = False,     
            do_sample: bool = True,
            temperature: float = 0.7,
            top_p: float = 0.9,
            max_new_tokens: int = 256,
            n: int = 3,     
            ):

        ## set general arguments
        self.finetuning = finetuning
        self.testing = testing
        self.inference = inference
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.mode = mode
        self.output_dir = output_dir
        self.load_adapter = load_adapter
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.seq_length = seq_length
        self.n = n
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.output_dir_fine_tuning = self.output_dir + '/sft'
        self.output_dir_test = self.output_dir + '/test'
        os.makedirs(self.output_dir_fine_tuning, exist_ok = True)
        os.makedirs(self.output_dir_test, exist_ok = True)

        ## set arguments for text generation
        self.gen_kwargs = dict(
            do_sample = do_sample,
            temperature = temperature,
            top_p = top_p,
            max_new_tokens = max_new_tokens)
        
        # 4-bit quantization config
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16")

        # tokenizer and model
        logger.info(f"loading tokenizer for {model_name}")
        if load_adapter:

            load_adapter_path = os.path.join(self.output_dir_fine_tuning, "adapter")
            peft_cfg = PeftConfig.from_pretrained(load_adapter_path)
            base_for_tokenizer = peft_cfg.base_model_name_or_path
            logger.info(f"set base model from adapter: {base_for_tokenizer}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_for_tokenizer,
                use_fast=True,
                trust_remote_code=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"loading base model: {base_for_tokenizer}")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_for_tokenizer,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            logger.info(f"loading PEFT adapter from: {load_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                load_adapter_path,
                is_trainable = False)

        else:

            base_for_tokenizer = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_for_tokenizer,
                use_fast=True,
                trust_remote_code=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        ## set mode 
        if self.mode == 'boring': from src.prompts.prompts_boring import SYSTEM_PROMPT, USER_TEMPLATE
        elif self.mode == 'trippy': from src.prompts.prompts_trippy import SYSTEM_PROMPT, USER_TEMPLATE
        elif self.mode == 'default': from src.prompts.prompts_default import SYSTEM_PROMPT, USER_TEMPLATE
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.USER_TEMPLATE = USER_TEMPLATE

        ## set the LoRA config for parameter efficient fine tuning (peft)
        logger.info(f"preparing LoRA config for {model_name}")
        self.peft_config = config_LoRA(
                        lora_r, 
                        lora_alpha, 
                        lora_dropout,
                        bias = "none", 
                        task_type = "CAUSAL_LM",
                        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])

        ## set supervised finetuning config
        logger.info(f"configuring SFT training with {model_name}")
        self.sft_config = config_SFT(
                        self.output_dir_fine_tuning,
                        seq_length,
                        packing = False,
                        per_device_train_batch_size = per_device_bs,
                        grad_accum = grad_accum,
                        learning_rate = lr,
                        num_train_epochs = epochs,
                        lr_scheduler_type = 'cosine',
                        warmup_ratio = 0.03,
                        logging_steps = logging_steps,
                        save_steps = save_steps,
                        save_total_limit = 2,
                        bf16 = False,
                        fp16 = True,                         
                        gradient_checkpointing = True,
                        optim = "paged_adamw_8bit",
                        report_to = 'none')
    
    def _formatting_func(self, example):
        q = example["question"]
        t = example.get("assistant_target", "") 

        if self.finetuning == True:
            msgs = [{"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.USER_TEMPLATE.format(question = q)},
                {"role": "assistant", "content": t}]
            chat = self.tokenizer.apply_chat_template(msgs, tokenize = False,
                                                    add_generation_prompt = False)
        if self.testing == True:
            msgs = [{"role":"system", "content": self.SYSTEM_PROMPT},
                {"role":"user", "content": self.USER_TEMPLATE.format(question = q)}]
            chat = self.tokenizer.apply_chat_template(msgs, tokenize = False,
                                                    add_generation_prompt = True)
        return chat
    
    def _cut_at_tag(self, s: str) -> str:
        end = s.find("</answer>")
        return s[:end + 9] if end != -1 else s
    
    def _normalize(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r"[,$]", "", s)
        s = re.sub(r"\s+", "", s)
        s = s.rstrip(".")
        return s
    
    def _extract_answer(self, text: str):
        num_pat = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
        tag = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S)
        if tag:
            return self._normalize(tag[-1])
        nums = re.findall(num_pat, text)
        if nums:
            return self._normalize(nums[-1])
        return None
    
    def _majority_vote(self, outputs):
        answer_to_text = {}
        valid_answers = []
        
        for output in outputs:
            answer = self._extract_answer(self._cut_at_tag(output))
            if answer is not None:
                valid_answers.append(answer)
                if answer not in answer_to_text:
                    answer_to_text[answer] = output
        
        if not valid_answers:
            return None, None
        
        most_common_answer = Counter(valid_answers).most_common(1)[0][0]
        return most_common_answer, answer_to_text[most_common_answer]
        
    def run(self, train_set, validation_set):

        ## set the trainer
        self.trainer = SFTTrainer(
            model = self.model,
            processing_class = self.tokenizer,
            train_dataset = train_set,
            eval_dataset = validation_set,
            peft_config = self.peft_config,
            args = self.sft_config,
            formatting_func = self._formatting_func)

        ## training
        logger.info(f"starting training with {self.model_name} on {self.dataset_name}")
        self.trainer.train()

        logger.info(f"saving model and tokenizer to {self.output_dir_fine_tuning}")
        self.trainer.model.save_pretrained(os.path.join(self.output_dir_fine_tuning, "adapter"))
        self.tokenizer.save_pretrained(self.output_dir_fine_tuning)
        
        logger.info(f"""Done.""")

    @torch.inference_mode()
    def test(self,
                dataset, 
                batch_size: int):
        
        logger.info(f"evaluating {self.model_name} on {self.dataset_name} - # problems: {len(dataset)}")
        self.model.eval()
        correct = 0
        b_range = range(0, len(dataset), batch_size)

        with open(self.output_dir_test + f'/test_results_sft_{self.load_adapter}', "w", encoding="utf-8") as f:
            
            for b_start in b_range:

                logger.info(f"processing batch {b_start // batch_size + 1} out of {len(b_range)}")

                ## get the current dataset batch
                b_start_time = time.time()
                b_end = min(b_start + batch_size, len(dataset))
                current_batch = dataset.select(range(b_start, b_end)) 
                # current_batch = dataset[b_start: b_end]

                ## get prompts for majority voting
                prompts = [self._formatting_func(ex) for ex in current_batch]
                prompts = [prompt for prompt in prompts for _ in range(self.n)]

                ## tokenize inputs
                inputs = self.tokenizer(
                        prompts,
                        return_tensors = "pt",
                        padding = True,
                        truncation = True,
                        max_length = self.seq_length)
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)

                ## generate outputs
                outputs = self.model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    pad_token_id = self.tokenizer.pad_token_id,
                    eos_token_id = self.tokenizer.eos_token_id,
                    **self.gen_kwargs)

                ## decode only the new tokens, process, evaluate correctness
                correct_batch = 0
                generated_tokens = outputs[:, inputs['input_ids'].shape[-1]:]
                for j, ex in enumerate(current_batch):
                    question_tokens = generated_tokens[j * self.n: (j * self.n) + self.n]
                    question_texts = self.tokenizer.batch_decode(question_tokens, skip_special_tokens = True)
                    texts = [text.strip() for text in question_texts]
                    pred_answer, pred_text = self._majority_vote(texts)

                    exact = False
                    if (pred_answer and is_correct(pred_answer, ex["gold_answer"])):
                        correct_batch += 1
                        correct += 1
                        exact = True

                    result = {"question": ex.get("question", ""),
                        "prediction_text": pred_text,
                        "predicted_answer": pred_answer,
                        "gold_answer": ex["gold_answer"],
                        "exact_match": exact,
                        "id": ex["id"], ## WHY THIS DOES NOT WORK?
                        "tasK_type": ex["task_type"]}
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    logger.info(f"DEBUG: final results \n {result}") ##DEBUG: REMOVE THIS LATER

                del inputs, outputs, generated_tokens

                batch_time = time.time() - b_start_time
                logger.info(f"total time: {batch_time / 60:.2f} minutes")
                logger.info(f"accuracy on batch: {100 * correct_batch  / len(current_batch):.1f}%")

        accuracy = correct / len(dataset)
        logger.info(f"evaluation complete. Correct answers: {correct} - Accuracy: {accuracy:.3f}")
        logger.info(f"inference results saved to: {self.output_dir_test}")
