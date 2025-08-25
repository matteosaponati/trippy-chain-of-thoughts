from typing import Dict
import json
import torch
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

from src.filters import is_correct

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

class GeneratorModel:
    
    def __init__(self,
            model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",   ## the checkpoint used as teaching model
            dataset_name: str = "gsm8k",                               ## the dataset 
            split: str = "train",                                      ## dataset split
            mode: str = "boring",                                      ## the mode "trippy" or "boring") for RL
            output_dir: str = "../dataset/",                           ## output directory
            load_in_4bit: bool = True,                                 ## work with 8bit quantized models
            do_sample = True,                                          ## flag variable for do_sample for generation
            seq_length: int = 1024,                                    ## max sequence length
            max_new_tokens: int = 256,                                 ## maximum numbers of tokens that the model can generate
            temperature: float = 0.9,                                  ## temperature for generation (keep high ~1)
            top_p: float = 0.9,                                        ## choose within 90% of the whole prob distribution
            ):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.split = split
        self.mode = mode
        self.output_dir = output_dir
        self.seq_length = seq_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        
        ## define tokenizer
        logger.info(f"loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast = True,
                trust_remote_code = False,
                padding_side = 'left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        ## define model
        logger.info(f"loading model {model_name} - 4-bit quantization? {load_in_4bit}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                device_map = 'auto',
                trust_remote_code = False,
                low_cpu_mem_usage = True,
                quantization_config = bnb)

        ## set mode 
        if self.mode == 'boring': from src.prompts.generation_prompts_boring import SYSTEM_PROMPT, USER_TEMPLATE
        elif self.mode == 'trippy': from src.prompts.generation_prompts_trippy import SYSTEM_PROMPT, USER_TEMPLATE
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.USER_TEMPLATE = USER_TEMPLATE

    def _formatting_func(self, example):
        q = example["question"]
        a = example["golden_answer"]
        msgs = [{"role":"system", "content": self.SYSTEM_PROMPT},
            {"role":"user", "content": self.USER_TEMPLATE.format(question = q, answer = a)}]
        chat = self.tokenizer.apply_chat_template(msgs, tokenize = False,
                                                add_generation_prompt = True)
        return chat
    
    def _parse_trippy(self, text: str):
        
        STRICT_RE = re.compile(
            r'^\s*<trip_before>\s*(.+?)\s*</trip_before>\s*'
            r'<answer>\s*(.+?)\s*</answer>\s*'
            r'<end>\s*(.+?)\s*</end>\s*', re.DOTALL)
        
        match = STRICT_RE.match(text)
        if match:
            trip_before, ans, ending = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            return trip_before, ans, ending
        
        if '<end>' in text and '</end>' not in text:
            text = text + '</end>'
            match = STRICT_RE.match(text)
            if match:
                trip_before, ans, ending = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                return trip_before, ans, ending
        
        logger.info(f"DEBUG: parsing failed for text, {text}")
        return None, None, None

    @torch.inference_mode()
    def generate(self,
            dataset, 
            batch_size: int):
        
        logger.info(f"starting dataset generation: target = {len(dataset)} items, batch_size = {batch_size}")
        self.model.eval()
        kept = 0
        id_count = 0
        b_range = range(0, len(dataset), batch_size)

        with open(self.output_dir + f'/gsm8k_{self.mode}_{self.split}.json', "w", encoding="utf-8") as f:
            
            for b_start in b_range:

                logger.info(f"processing batch {b_start // batch_size + 1} out of {len(b_range)}")

                ## get the current dataset batch
                b_start_time = time.time()
                b_end = min(b_start + batch_size, len(dataset))
                current_batch = dataset.select(range(b_start, b_end)) 
                prompts = [self._formatting_func(ex) for ex in current_batch]

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
                    id_count += 1
                    text = self.tokenizer.decode(generated_tokens[j], skip_special_tokens=True)
                    trip_before, answer, end = self._parse_trippy(text.strip())

                    ## save only the correct ones
                    if (answer and is_correct(answer, ex["golden_answer"])):
                        correct_batch += 1
                        kept += 1
                        result = {
                                "id": id_count,
                                "split": self.split,
                                "question": ex["question"],
                                "gold_answer": ex["golden_answer"],
                                "trip_before": trip_before,
                                "answer": answer,
                                "end": end,
                                "assistant_answer": text}
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                del inputs, outputs, generated_tokens

                batch_time = time.time() - b_start_time
                logger.info(f"total time: {batch_time / 60:.2f} minutes")
                logger.info(f"success rate: {100 * correct_batch  / len(current_batch):.1f}%")
                logger.info(f"progress: {kept}/{len(dataset)} total items kept")     

        logger.info(f"synthesis complete. Kept {kept} items → saved to {self.output_dir}")
        logger.info(f"overall success rate: {100 * kept / len(dataset):.1f}%")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device) if hasattr(self.model, 'device') else self.device,
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'unknown')
        }