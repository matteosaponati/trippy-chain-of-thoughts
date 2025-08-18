from typing import List, Dict, Optional
import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

from src.synthetic_data.filters import is_parsed, is_correct, is_length

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

class DataGenerator:
    
    def __init__(
            self,
            model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
            dataset_name: str = "gsm8k",
            mode: str = "boring",
            torch_dtype: Optional[torch.dtype] = None,
            trust_remote_code: bool = False,
            load_in_8bit: bool = False,
            max_new_tokens: int = 384,
            temperature: float = 0.9,
            top_p: float = 0.9,
            n: int = 3,
            stop_sequences: Optional[List[str]] = None,
            repetition_penalty: float = 1.0):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop_sequences = stop_sequences
        self.repetition_penalty = repetition_penalty

        logger.info(f"loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast = True,
                trust_remote_code = trust_remote_code)
        logger.info(f"tokenizer loaded with vocab size: {self.tokenizer.vocab_size}")
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"loading model {model_name}")
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": "auto",
            "low_cpu_mem_usage": True}

        if load_in_8bit:
            logger.info("loading model in **8-bit quantized mode** (bitsandbytes)")
            quant_config = BitsAndBytesConfig(
                load_in_8bit = True)                
            model_kwargs["quantization_config"] = quant_config

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info(f"model loaded successfully on device(s): {self.model.device if hasattr(self.model, 'device') else 'multiple'}")

        if self.mode == 'boring':
            from src.prompts.prompts_boring import SYSTEM_PROMPT, USER_TEMPLATE
        elif self.mode == 'trippy':
            from src.prompts.prompts_trippy import SYSTEM_PROMPT, USER_TEMPLATE
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.USER_TEMPLATE = USER_TEMPLATE

    def _to_messages(self, question: str, answer: str) -> List[Dict]:
        """Convert a question string into a list of messages in chat format."""
        return [{"role":"system", "content": self.SYSTEM_PROMPT},
            {"role":"user", "content": self.USER_TEMPLATE.format(question = question, answer = answer)}]
    
    def _to_chat(self, messages: List[Dict]) -> str:
        """Takes a list of dictionary (each representing a prompt in chat format)
        and returns a formatted string or set of tokens."""
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize = False, 
            add_generation_prompt = True)

    def _self_consistency(self, outputs: List[str], 
                            gold_answer: str) -> Optional[tuple]:
        """Select the best output based on self-consistency."""
        if not outputs or not gold_answer:
            return None

        best = None
        best_score = float("-inf")

        for i, text in enumerate(outputs or []):
            
            trip_before, answer, end = is_parsed(text or "")
            if not (trip_before and answer and end):
                continue
            if not is_length(trip_before): 
                continue
            if not is_length(end): 
                continue
            if not is_correct(gold_answer, answer): 
                continue
            
            score = len(trip_before) + 0.5 * len(end)
            if score > best_score:
                best_score = score
                best = (trip_before, answer, end, text)
    
        return best

    def _to_wrapped_data(self, results, ex, dataset_name):
        trip_before, answer, end = results[:3]
        return {"messages": [{"role": "user","content": f"<problem>\n{ex['question']}\n</problem>"},
            {"role": "assistant","content": (f"<trip_before>{trip_before}</trip_before>\n"
                f"<answer>{answer}</answer>\n"f"<end>{end}</end>")}],
            "meta": {"source": dataset_name, "id": ex["uid"], "task_type": ex["task_type"]}}
        
    @torch.inference_mode()
    def generate(self, questions_batch: List[Dict], answers_batch: List[Dict]) -> List[str]:
        """Generate responses from the model."""

        generation_start = time.time()

        messages_batch = [self._to_messages(q, a) for q, a in zip(questions_batch, answers_batch)]
        prompts = [self._to_chat(messages) for messages in messages_batch]

        all_prompts = []
        for prompt in prompts:
            all_prompts.extend([prompt] * self.n)

        inputs = self.tokenizer(
            all_prompts, 
            return_tensors = "pt", 
            padding = True, 
            truncation = True,
            max_length = self.model.config.max_position_embeddings - self.max_new_tokens
            ).to(self.model.device)
    
        outputs = self.model.generate(**inputs,
                    do_sample = self.temperature > 0,
                    temperature = self.temperature if self.temperature > 0 else None,
                    top_p = self.top_p if self.temperature > 0 else None,
                    max_new_tokens = self.max_new_tokens,
                    repetition_penalty = self.repetition_penalty)

        input_token_length = inputs['input_ids'].shape[-1]
        generated_tokens = outputs[:, input_token_length:]
        texts = self.tokenizer.batch_decode(generated_tokens, 
                    skip_special_tokens = True)
        
        processed_texts = []
        for text in texts:
            generated = text.strip()
            if self.stop_sequences:
                for stop_seq in self.stop_sequences:
                    if stop_seq in generated:
                        generated = generated.split(stop_seq)[0].strip()
                        break
            processed_texts.append(generated)

        results = []
        for i in range(len(questions_batch)):
            start_idx = i * self.n
            end_idx = start_idx + self.n
            results.append(processed_texts[start_idx:end_idx])

        del inputs, outputs, generated_tokens
        
        total_generation_time = time.time() - generation_start
        logger.info(f"total generation: {total_generation_time:.3f}s")
        logger.info(f"generation rate: {len(all_prompts) / total_generation_time:.2f} prompts/sec")

        return results
    
    def synthesize(self, dataset, 
                out_path, 
                limit = None,
                batch_size = 4):

        kept = 0
        total = len(dataset) if not limit else min(limit, len(dataset))
        logger.info(f"starting synthesis: target={total} items, batch_size={batch_size}")

        with open(out_path, "w", encoding = "utf-8") as f:
            
            idx_range = len(range(0, len(dataset), batch_size))
            for batch_start in range(0, len(dataset), batch_size):

                batch_start_time = time.time()

                batch_end = min(batch_start + batch_size, len(dataset))
                if limit:
                    batch_end = min(batch_end, batch_start + (limit - kept))
                current_batch = dataset[batch_start:batch_end]
                questions = [ex["question"] for ex in current_batch]
                answers = [ex["gold_answer"] for ex in current_batch]
                logger.info(f"processing batch {batch_start // batch_size + 1} out of {idx_range} ({len(current_batch)} items)")

                batch_results = self.generate(questions, answers)
                
                batch_kept = 0
                for ex, results in zip(current_batch, batch_results):

                    best_results = self._self_consistency(results, ex["gold_answer"])
                    if not best_results: continue
                    item = self._to_wrapped_data(best_results, ex, self.dataset_name)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    kept += 1
                    batch_kept += 1

                batch_time = time.time() - batch_start_time
                logger.info(f"total time: {batch_time / 60:.2f} minutes")
                logger.info(f"success rate: {100 * batch_kept / len(current_batch):.1f}%")
                logger.info(f"progress: {kept}/{total} total items kept")

        logger.info(f"synthesis complete. Kept {kept} items → saved to {out_path}")
        logger.info(f"overall success rate: {100 * kept / total:.1f}%")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device) if hasattr(self.model, 'device') else self.device,
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'unknown')
        }