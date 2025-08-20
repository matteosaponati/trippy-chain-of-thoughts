from typing import List, Dict, Optional
import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

from src.filters import self_consistency, evaluate_answer, majority_vote, is_correct
from src.chat_utils import to_messages, to_chat, to_wrapped_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

class GeneratorTeacherModel:
    
    def __init__(self,
            model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",   ## the checkpoint used as teaching model
            dataset_name: str = "gsm8k",                               ## the dataset 
            mode: str = "boring",                                      ## the mode I"trippy" or "boring") for RL
            load_in_8bit: bool = True,                                 ## work with 8bit quantized models
            max_new_tokens: int = 384,                                 ## maximum numbers of tokens that the model can generate
            temperature: float = 0.9,                                  ## temperature for generation (keep high >1)
            top_p: float = 0.9,                                        ## choose within 90% of the whole prob distribution
            n: int = 3,                                                ## number of repetitions per generation (keep best out of n)                
            repetition_penalty: float = 1.0                            ## avoid extensive repetitions in generation
            ):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.repetition_penalty = repetition_penalty
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        ## define tokenizer
        logger.info(f"loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast = True,
                trust_remote_code = False,
                padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        ## define model
        logger.info(f"loading model {model_name} - 8-bit quantization? {load_in_8bit}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                device_map = 'auto',
                trust_remote_code = False,
                low_cpu_mem_usage = True,
                quantization_config = BitsAndBytesConfig(load_in_8bit = True))
        logger.info(f"model loaded successfully on device(s): {self.model.device if hasattr(self.model, 'device') else 'multiple'}")

        ## set mode 
        if self.mode == 'boring': from src.prompts.prompts_boring import SYSTEM_PROMPT, USER_TEMPLATE
        elif self.mode == 'trippy': from src.prompts.prompts_trippy import SYSTEM_PROMPT, USER_TEMPLATE
        elif self.mode == 'evaluate': from src.prompts.prompts_evaluate import SYSTEM_PROMPT, USER_TEMPLATE
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.USER_TEMPLATE = USER_TEMPLATE

    @torch.inference_mode()
    def generate(self, 
                questions_batch: List[Dict], 
                answers_batch: List[Dict]) -> List[str]:
        """Generate responses from the model."""

        generation_start = time.time()

        ## get question and answer, convert to chat, and extend it for multiple choices
        messages_batch = [to_messages(q, a, self.SYSTEM_PROMPT, self.USER_TEMPLATE, self.mode) for q, a in zip(questions_batch, answers_batch)]
        prompts = [to_chat(messages, self.tokenizer) for messages in messages_batch]
        prompts = [prompt for prompt in prompts for _ in range(self.n)]

        inputs = self.tokenizer(prompts, 
                return_tensors = "pt",                                                          ## return torch tensors
                padding = True,                                                                 ## do padding to longest prompt
                truncation = True,                                                              ## truncate prompts to respect max_length
                max_length = self.model.config.max_position_embeddings - self.max_new_tokens    ## leave space for new tokens
                ).to(self.model.device)
    
        outputs = self.model.generate(**inputs,
                    do_sample = self.temperature > 0,
                    temperature = self.temperature if self.temperature > 0 else None,
                    top_p = self.top_p if self.temperature > 0 else None,
                    max_new_tokens = self.max_new_tokens,
                    repetition_penalty = self.repetition_penalty)

        ## decode only the new tokens and process it
        generated_tokens = outputs[:, inputs['input_ids'].shape[-1]:]
        texts = []
        for i in range(len(questions_batch)):
            question_tokens = generated_tokens[i * self.n: (i * self.n) + self.n]
            question_texts = self.tokenizer.batch_decode(question_tokens, skip_special_tokens = True)
            texts.append([text.strip() for text in question_texts])

        del inputs, outputs, generated_tokens
        
        total_generation_time = time.time() - generation_start
        logger.info(f"total generation: {total_generation_time:.3f}s")
        logger.info(f"generation rate: {len(prompts) / total_generation_time:.2f} prompts/sec")

        return texts
    
    def synthesize(self, 
                dataset, 
                out_path: str, 
                batch_size: int):

        logger.info(f"starting dataset generation: target = {len(dataset)} items, batch_size = {batch_size}")
        b_range = range(0, len(dataset), batch_size)
        kept = 0

        with open(out_path, "w", encoding = "utf-8") as f:
            
            for b_start in b_range:
                
                logger.info(f"processing batch {b_start // batch_size + 1} out of {len(b_range)}")

                ## get the current dataset batch
                b_start_time = time.time()
                b_end = min(b_start + batch_size, len(dataset))
                current_batch = dataset[b_start: b_end]

                ## generate results for the given batch
                questions = [ex["question"] for ex in current_batch]
                answers = [ex["gold_answer"] for ex in current_batch]
                batch_results = self.generate(questions, answers)

                ## pick best results and save 
                best_results = []
                for ex, results in zip(current_batch, batch_results):
                    best_results.append(self_consistency(results, ex["gold_answer"]))
                wrapped_items = to_wrapped_data(best_results, current_batch, self.dataset_name)
                f.writelines(json.dumps(item, ensure_ascii = False) + "\n" for item in wrapped_items)   

                kept += len(wrapped_items)       
                batch_time = time.time() - b_start_time
                logger.info(f"total time: {batch_time / 60:.2f} minutes")
                logger.info(f"success rate: {100 * len(wrapped_items)  / len(current_batch):.1f}%")
                logger.info(f"progress: {kept}/{len(dataset)} total items kept")     

        logger.info(f"synthesis complete. Kept {kept} items → saved to {out_path}")
        logger.info(f"overall success rate: {100 * kept / len(dataset):.1f}%")

    def evaluate(self,
                dataset, 
                batch_size: int):
        
        logger.info(f"evaluating {self.model_name} on {self.dataset_name} - # problems: {len(dataset)}")
        correct = 0
        b_range = range(0, len(dataset), batch_size)

        for b_start in b_range:

            logger.info(f"processing batch {b_start // batch_size + 1} out of {len(b_range)}")

            ## get the current dataset batch
            b_start_time = time.time()
            b_end = min(b_start + batch_size, len(dataset))
            current_batch = dataset[b_start: b_end]

            ## generate results for the given batch
            questions = [ex["question"] for ex in current_batch]
            answers = [ex["gold_answer"] for ex in current_batch]
            batch_results = self.generate(questions, answers)

            # evaluate each question in batch
            correct_batch = 0
            for j, ex in enumerate(current_batch):
                responses = batch_results[j]  
                response = majority_vote(responses)
                if (response and is_correct(response, ex["gold_answer"])):
                    correct_batch += 1
                    correct += 1

            batch_time = time.time() - b_start_time
            logger.info(f"total time: {batch_time / 60:.2f} minutes")
            logger.info(f"accuracy on batch: {100 * correct_batch  / len(current_batch):.1f}%")

        accuracy = correct / len(dataset)
        logger.info(f"evaluation complete. Correct answers: {correct} - Accuracy: {accuracy:.3f}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device) if hasattr(self.model, 'device') else self.device,
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'unknown')
        }