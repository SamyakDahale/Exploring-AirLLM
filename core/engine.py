"""
Engine module for InBox Lens.
Handles AirLLM initialization, tokenization, generation, context windowing, and logging.
"""

import os
import time
import logging
import torch
from airllm import AutoModel
from .prompts import format_prompt

# Setup basic logging system
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AirLLMEngine")

class AirLLMEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_context_length = 2048 # TinyLlama max context size
        
    def load(self):
        if self.model is not None:
            return
            
        if not os.path.exists(self.model_path):
            logger.error(f"Model path not found: {self.model_path}")
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
        logger.info(f"Loading AirLLM model from {self.model_path}")
        logger.info(f"Target device identified as: {self.device}")
        
        try:
            # --- AIRLLM SHARDING EXPLANATION ---
            # Under the hood, AutoModel.from_pretrained uses layer-wise model sharding.
            # Instead of loading the entire multi-gigabyte model into VRAM/RAM all at once,
            # AirLLM keeps the model weights on disk and only proxies a few layers into memory 
            # at a time during the forward pass. This guarantees we can run very large models 
            # on completely constrained hardware efficiently.
            self.model = AutoModel.from_pretrained(self.model_path)
            logger.info("Model loaded successfully into AirLLM abstraction.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def generate_response(self, system_message: str, user_prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        """
        Takes formatted prompts, handles tokenization & context windowing, and generates the output.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
            
        formatted_prompt = format_prompt(system_message, user_prompt)
        
        # --- CONTEXT WINDOWING ---
        # Calculate safe max length for inputs to leave room for the generated tokens.
        # This prevents crashing TinyLlama's strict 2,048 limit.
        safe_input_length = self.max_context_length - max_new_tokens
        
        start_time = time.time()
        
        try:
            # Tokenize input. 
            # We enforce truncation=True and max_length here.
            # Truncation automatically limits the string to `safe_input_length` tokens.
            input_tokens = self.model.tokenizer(
                [formatted_prompt],
                return_tensors="pt",
                return_attention_mask=False,
                truncation=True,
                max_length=safe_input_length, 
                padding=False
            )
            
            input_ids = input_tokens['input_ids']
            
            if self.device == "cuda":
                input_ids = input_ids.cuda()
                
            logger.info(f"Starting generation (Input tokens: {input_ids.shape[1]})...")
            
            generation_output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                temperature=temperature,
                top_p=0.9
            )
            
            output_text = self.model.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f} seconds.")
            
            # Strip out the prompt to leave just the assistant's part
            assistant_tag = "<|assistant|>\n"
            if assistant_tag in output_text:
                return output_text.split(assistant_tag)[-1].strip()
                
            return output_text.strip()
            
        except torch.cuda.OutOfMemoryError as oom:
            logger.error("CUDA Out Of Memory error encountered during generation!", exc_info=True)
            return "Error: Out of Memory (VRAM/RAM). The input might be too large despite sharding."
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            return f"Error during generation: {e}"
