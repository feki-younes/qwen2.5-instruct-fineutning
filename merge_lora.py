#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to merge LoRA adapters with a quantized base model.
"""
import os
import logging
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc

# === Configuration ===
BASE_MODEL_NAME = "Qwen2.5-3B-Instruct-unsloth-bnb-4bit" # Or the path to your base model if local
ADAPTERS_PATH = "./qwen_finetuned_unsloth/checkpoint-700/" # Path where your fine-tuned adapters are saved
OUTPUT_MERGED_PATH = "./Qwen2.5-3B-Instruct-Audit-Reporter-4bit" # Path to save the merged model

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_lora_to_quantized_model():
    """Loads a quantized base model, merges LoRA adapters, and saves the result."""
    try:
        logger.info("Starting merge process...")

        # 1. Load the Base Quantized Model
        logger.info(f"Loading base quantized model: {BASE_MODEL_NAME}")
        # Load the base model. Ensure quantization config is correctly loaded.
        # Using torch_dtype=torch.bfloat16 or torch.float16 is common for quantized models
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto", # Adjust device_map if needed (e.g., "cpu" for merging on CPU, but very slow)
            # The quantization config should ideally be part of the saved base model.
            # If loading from a local path that was saved with bnb config, it might be loaded automatically.
            # If loading from HF hub name, ensure it points to the correct quantized version.
        )
        logger.info("Base quantized model loaded.")

        # 2. Load the Tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME) # Usually the same as base model
        logger.info("Tokenizer loaded.")

        # 3. Load the LoRA Adapters
        logger.info(f"Loading LoRA adapters from: {ADAPTERS_PATH}")
        # Load the PEFT model (LoRA adapters) on top of the base model
        merged_model = PeftModel.from_pretrained(base_model, ADAPTERS_PATH, torch_dtype=base_model.dtype)
        logger.info("LoRA adapters loaded.")

        # 4. Merge Adapters
        logger.info("Merging adapters into the base model...")
        # Merge the LoRA weights into the base model weights
        # This operation modifies the base model in-place conceptually within the PeftModel wrapper
        merged_model = merged_model.merge_and_unload() # This returns the underlying base model with weights merged
        logger.info("Adapters merged successfully.")

        # Clean up
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5. Save the Merged Model and Tokenizer
        logger.info(f"Saving merged model and tokenizer to: {OUTPUT_MERGED_PATH}")
        output_path = Path(OUTPUT_MERGED_PATH)
        output_path.mkdir(parents=True, exist_ok=True)

        merged_model.save_pretrained(output_path, safe_serialization=True) # safe_serialization is recommended
        tokenizer.save_pretrained(output_path)
        logger.info("Merged model and tokenizer saved successfully.")

        logger.info("Merge process completed!")
        logger.info(f"Merged model is located at: {OUTPUT_MERGED_PATH}")

    except Exception as e:
        logger.error(f"An error occurred during merging: {e}")
        raise

if __name__ == "__main__":
    merge_lora_to_quantized_model()