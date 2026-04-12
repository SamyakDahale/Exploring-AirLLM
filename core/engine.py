"""
Engine module for InBox Lens.
Handles model initialization natively (Transformers), tokenization, and generation.
Dumbed-down to remove Optimum and AirLLM dependencies while maintaining local offline constraints.
"""

import os
import time
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompts import format_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLMEngine")

class AirLLMEngine:
    # Kept the class name AirLLMEngine for compatibility with the UI code, 
    # but under the hood, we are now using native transformers architecture.
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_context_length = 2048
        
    def load(self):
        if self.model is not None:
            return
            
        if not os.path.exists(self.model_path):
            logger.error(f"Model path not found: {self.model_path}")
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
        logger.info(f"Loading native Transformers model from {self.model_path}")
        logger.info(f"Target device: {self.device}")
        
        try:
            # We enforce standard transformers loading.
            # With `device_map="auto"` (via accelerate), it will automatically manage VRAM/RAM 
            # safely without needing optimum or airllm explicitly.
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # If we don't have CUDA, we explicitly ensure the model rests on CPU
            if self.device == "cpu":
                self.model.to("cpu")
                
            logger.info("Model loaded successfully into Native abstraction.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def generate_response(self, system_message: str, user_prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
            
        formatted_prompt = format_prompt(system_message, user_prompt)
        safe_input_length = self.max_context_length - max_new_tokens
        start_time = time.time()
        
        try:
            # Tokenize natively
            input_tokens = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=safe_input_length, 
                padding=False
            )
            
            input_ids = input_tokens['input_ids'].to(self.model.device)
            attention_mask = input_tokens['attention_mask'].to(self.model.device)
            logger.info(f"Starting generation (Tokens: {input_ids.shape[1]})...")
            
            generation_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Output decoding
            # Slices the output by the length of the input so we only return the newly generated text
            input_length = input_ids.shape[1]
            generated_tokens = generation_output[0][input_length:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f} seconds.")
            
            return output_text.strip()
            
        except torch.cuda.OutOfMemoryError as oom:
            logger.error("CUDA Out Of Memory error encountered during generation!", exc_info=True)
            return "Error: Out of Memory (VRAM). The input might be too large."
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            return f"Error during generation: {e}"
