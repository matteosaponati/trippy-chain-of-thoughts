from typing import List, Dict, Optional
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

from src.synthetic_data.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.synthetic_data.filters import parse_output, is_correct, length_ok

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

class DataGenerator:
    
    def __init__(
            self,
            model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
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
        logger.info(f"tokenizer loaded. Vocab size: {self.tokenizer.vocab_size}")
        
        logger.info(f"loading model {model_name}")
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": "auto",
            "low_cpu_mem_usage": True}
        if load_in_8bit:
            logger.info("loading model in **8-bit quantized mode** (bitsandbytes)")
            quant_config = BitsAndBytesConfig(
                load_in_8bit = True
            )
            model_kwargs["quantization_config"] = quant_config
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info(f"model loaded successfully on device(s): {self.model.device if hasattr(self.model, 'device') else 'multiple'}")
    
    def _to_messages(self, question: str):
        return [
            {"role":"system", "content": SYSTEM_PROMPT},
            {"role":"user", "content": USER_TEMPLATE.format(question = question)}]
    
    def _to_chat(self, messages: List[Dict]) -> str:
        """Takes a list of dictionary (each representing a prompt in chat format)
        and returns a formatted string or set of tokens."""
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize = False, 
            add_generation_prompt = True)
    
    def _pick_best(self, outputs, gold_answer):
        best = None
        best_len = -1
        for i, text in enumerate(outputs):
            trip, final = parse_output(text or "")
            if not (trip and final): 
                # logger.info("failed: missing trip or final")
                continue
            if not length_ok(trip): 
                # logger.info("failed: length check")
                continue
            if not is_correct(gold_answer, final): 
                # logger.info("failed: incorrect answer")
                continue
            L = len(trip)
            if L > best_len:
                best_len = L
                best = (trip, final, text)
            return best
    
    def _to_wrapped_data(self, results, ex):
        trip, final, _ = results
        return {"messages":[
                {"role":"user", "content": f"<problem>\n{ex['question']}\n</problem>"},
                {"role":"assistant", "content": f"<trip>{trip}</trip>\nFinal: {final}"}],
                "meta":{"source": "gsm8k", "id": ex["uid"], "task_type": ex["task_type"]}}
        
    @torch.inference_mode()
    def generate(self, 
                 questions_batch: List[Dict]) -> List[str]:
        """Generate responses from the model."""

        messages_batch = [self._to_messages(q) for q in questions_batch]
        prompts = [self._to_chat(messages) for messages in messages_batch]
        all_prompts = []
        for prompt in prompts:
            all_prompts.extend([prompt] * self.n)

        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            all_prompts, 
            return_tensors = "pt", 
            padding = True, 
            truncation = True,
            max_length = self.model.config.max_position_embeddings - self.max_new_tokens
            ).to(self.model.device)

        input_token_length = inputs['input_ids'].shape[-1]
    
        outputs = self.model.generate(**inputs,
                    do_sample = self.temperature > 0,
                    temperature = self.temperature if self.temperature > 0 else None,
                    top_p = self.top_p if self.temperature > 0 else None,
                    max_new_tokens = self.max_new_tokens,
                    repetition_penalty = self.repetition_penalty)

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

        return results
    
    def synthesize(self, dataset, 
                out_path, 
                limit = None,
                batch_size = 4):

        kept = 0
        total = len(dataset) if not limit else min(limit, len(dataset))
        logger.info(f"starting synthesis: target={total} items, batch_size={batch_size}")

        with open(out_path, "w", encoding = "utf-8") as f:
            
            idx = 0 
            idx_range = len(range(0, len(dataset), batch_size))
            for batch_start in range(0, len(dataset), batch_size):

                logger.info(f" batch # {idx} out of {idx_range}")

                batch_end = min(batch_start + batch_size, len(dataset))
                if limit:
                    batch_end = min(batch_end, batch_start + (limit - kept))
                current_batch = dataset[batch_start:batch_end]
                questions = [ex["question"] for ex in current_batch]
                logger.info(f"processing batch {batch_start // batch_size + 1} ({len(current_batch)} items)")

                batch_results = self.generate(questions)
                
                for ex, results in zip(current_batch, batch_results):

                    # if kept >= limit:
                    #     break    
                    # logger.info(f"[{kept+1}/{total}] processing example ID = {ex.get('uid', batch_start)}")

                    best_results = self._pick_best(results, ex["gold_answer"])
                    if not best_results:
                        logger.warning(f"no valid output for example ID = {ex.get('uid', batch_start)}. skipping.")
                        continue
                    
                    item = self._to_wrapped_data(best_results, ex)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    kept += 1

        logger.info(f"synthesis complete. Kept {kept} items → saved to {out_path}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device) if hasattr(self.model, 'device') else self.device,
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'unknown')
        }