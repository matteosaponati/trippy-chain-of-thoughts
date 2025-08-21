import os
import re
import time
import json
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import PeftConfig, AutoPeftModelForCausalLM, PeftModel
import torch
import logging

import numpy as np
from sklearn.metrics import accuracy_score

from src.filters import is_correct
from src.ft_utils import config_LoRA, config_SFT
from src.chat_utils import to_messages, to_chat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

class TrainerTrippyModel:

    def __init__(self,
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
            repetition_penalty: float = 1.1,
            n: int = 3,     
            ):

        ## set general arguments
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
            max_new_tokens = max_new_tokens,
            repetition_penalty = repetition_penalty)
        
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
        elif self.mode == 'evaluate': from src.prompts.prompts_evaluate import SYSTEM_PROMPT, USER_TEMPLATE
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
        a = example.get("gold_answer", "")

        msgs = to_messages(question = q,answer = a, 
                           SYSTEM_PROMPT = self.SYSTEM_PROMPT,
                           USER_TEMPLATE = self.USER_TEMPLATE,
                           mode = self.mode)
        return to_chat(msgs, self.tokenizer, add_generation_prompt = False)
    
    def extract_answer(self, text): ## make this more simple

        patterns = [r"<answer>\s*(.*?)\s*</answer>",
                    r"[Tt]he answer is[:\s]*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
                r"[Aa]nswer[:\s]*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)"]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                answer = matches[-1].replace(",", "")
                try:
                    return answer
                except:
                    continue
        return None
    
    def majority_vote(self, outputs): ## make this more simple
        
        answer_to_text = {} 
        answers = []
        
        for output in outputs:
            answer = self.extract_answer(output)
            if answer is not None:
                answers.append(answer)
                if answer not in answer_to_text:
                    answer_to_text[answer] = output
        
        if not answers:
            return None, None
        
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)
        if most_common:
            majority_answer = most_common[0][0]
            associated_text = answer_to_text[majority_answer]
            return (majority_answer, associated_text)
        else:
            return None, None
        
    def run(self, train_set, validation_set):

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
        results = []
        correct = 0
        b_range = range(0, len(dataset), batch_size)

        for b_start in b_range:

            logger.info(f"processing batch {b_start // batch_size + 1} out of {len(b_range)}")

            ## get the current dataset batch
            b_start_time = time.time()
            b_end = min(b_start + batch_size, len(dataset))
            current_batch = dataset[b_start: b_end]

            ## get prompts for majority voting (check this)
            questions = [ex["question"] for ex in current_batch]
            answers = [ex["gold_answer"] for ex in current_batch]
            messages_batch = [to_messages(q, a, self.SYSTEM_PROMPT, self.USER_TEMPLATE, self.mode) for q, a in zip(questions, answers)]
            prompts = [to_chat(messages, self.tokenizer, add_generation_prompt = True) for messages in messages_batch] ## I think Ture is good for inference?
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
                pred_answer, pred_text = self.majority_vote(texts)
                
                exact = False
                if (pred_answer and is_correct(pred_answer, ex["gold_answer"])):
                    correct_batch += 1
                    correct += 1
                    exact = True

                results.append({
                    "question": ex.get("question", ""),
                    "prediction_text": pred_text,
                    "predicted_answer": pred_answer,
                    "gold_answer": ex["gold_answer"],
                    "exact_match": exact})

            del inputs, outputs, generated_tokens

            batch_time = time.time() - b_start_time
            logger.info(f"total time: {batch_time / 60:.2f} minutes")
            logger.info(f"accuracy on batch: {100 * correct_batch  / len(current_batch):.1f}%")

        accuracy = correct / len(dataset)
        logger.info(f"evaluation complete. Correct answers: {correct} - Accuracy: {accuracy:.3f}")
        
        with open(self.output_dir_test + f'/test_results_sft_{self.load_adapter}', "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"inference results saved to: {self.output_dir_test}")

    # @staticmethod
    # def _extract(tag_re, text):
    #     m = tag_re.search(text or "")
    #     return m.group(1).strip() if m else ""

    # @staticmethod
    # def _strip(s):
    #     return (s or "").strip()

    # def _formatting_func(self, example):

    #     msgs = example.get("messages", [])
    #     user_msg = next((m for m in msgs if m.get("role") == "user"), None)
    #     asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)

    #     problem = self._strip(user_msg.get("content") if user_msg else example.get("question", ""))
    #     asst_text = asst_msg.get("content") if asst_msg else ""

    #     _ANS_RE  = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
    #     _TRIP_RE = re.compile(r"<trip_before>\s*(.*?)\s*</trip_before>", re.IGNORECASE | re.DOTALL)
    #     _END_RE  = re.compile(r"<end>\s*(.*?)\s*</end>", re.IGNORECASE | re.DOTALL)

    #     # pull structured pieces
    #     trip = self._extract(_TRIP_RE, asst_text)
    #     ans  = self._extract(_ANS_RE,  asst_text) or self._strip(example.get("gold_answer", ""))
    #     end  = self._extract(_END_RE,  asst_text)

    #     # Build prompt: system + user (problem + rationale tags)
    #     user_prompt_parts = [problem]
    #     if trip:
    #         user_prompt_parts.append(f"<trip_before>{trip}</trip_before>")
    #     if end:
    #         user_prompt_parts.append(f"<end>{end}</end>")
    #     user_prompt = "\n".join(p for p in user_prompt_parts if p).strip()

    #     messages = [
    #         {"role": "system",   "content": getattr(self, "SYSTEM_PROMPT", "")},
    #         {"role": "user",     "content": user_prompt},
    #         {"role": "assistant","content": f"<answer>{ans}</answer>"},
    #     ]

    #     return self.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=False)
