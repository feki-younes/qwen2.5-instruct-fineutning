#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced LLM Fine-tuning Script using Unsloth with Tokenizer-Based Filtering
This script implements a production-ready fine-tuning pipeline for Qwen2.5 models
leveraging Unsloth for speed and memory efficiency. It uses a masked loss strategy
where only the assistant's responses contribute to the training loss, ensuring
efficient instruction tuning.
Key Features:
- Utilizes Unsloth for 4-bit Quantized LoRA (QLoRA) fine-tuning.
- Implements masked loss computation for optimal training on response tokens.
- Integrates with MLflow for experiment tracking.
- Maintains structured logging and configuration management.
- Filters data using the tokenizer for accurate sequence length and structure checks.
- Designed for professional deployment and reproducibility.
Version: 2.1 - Tokenizer-Based Data Filtering
"""
import os
import sys
import json
import logging
import traceback
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from datasets import Dataset as HFDataset, DatasetDict
from langdetect import detect
from tqdm.auto import tqdm
import mlflow
import mlflow.pytorch

# === Unsloth Import ===
try:
    # Unsloth provides significantly faster and more memory-efficient training
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    # Unsloth's trainer automatically handles masked loss for responses
    from trl import SFTTrainer 
    from transformers import TrainingArguments
    # Use Unsloth's chat template utilities
    from unsloth.chat_templates import get_chat_template 
except ImportError as e:
    print(f"Error importing Unsloth: {e}")
    print("Please install Unsloth: `pip install 'unsloth[cu121_ampere] @ git+https://github.com/unslothai/unsloth.git'`")
    sys.exit(1)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix logging configuration
def setup_logging():
    """Setup logging with proper configuration for both console and file output."""
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('training_unsloth_token_filtered.log', mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

# ============================================================================
# Configuration Section
# ============================================================================
@dataclass
class TrainingConfig:
    """
    Configuration for LLM fine-tuning using Unsloth with QLoRA.
    This configuration uses Unsloth's optimized methods for speed and memory
    efficiency, along with a masked loss strategy focusing on assistant responses.
    """
    # --- Model Configuration ---
    model_name: str = "Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
    """str: Name of the pre-trained model on Hugging Face Hub."""
    output_dir: str = "./qwen_finetuned_unsloth"
    """str: Directory to save the fine-tuned model and tokenizer."""
    # --- Data Paths ---
    train_path: str = "./data/full_training.json"
    val_path: str = "./data/full_validation.json"
    test_path: str = "./data/full_test.json"
    # --- Task Configuration ---
    selected_tasks: List[str] = field(default_factory=lambda: ["reco_gen", "synth_gen_interm", "synth_gen_final"])
    """List[str]: The specific task types to filter data by."""
    # --- Data Filtering Parameters (Tokenizer-Based) ---
    # These control the filtering of samples using the tokenizer
    min_assistant_response_tokens: int = 20 # Minimum tokens for assistant response
    """int: Minimum number of tokens for an assistant response to be considered valid."""
    # --- Unsloth-Specific Parameters ---
    # These parameters control the quantization and LoRA setup via Unsloth
    load_in_4bit: bool = True
    """bool: Enable 4-bit quantization for memory efficiency (QLoRA)."""
    dtype: Optional[torch.dtype] = None # Let Unsloth auto-detect based on hardware
    """Optional[torch.dtype]: Data type for model weights. None lets Unsloth choose (bfloat16 if supported, else float16)."""
    # LoRA Configuration - Optimized for instruction tuning with Unsloth defaults
    max_seq_length: int = 7168 # Unsloth supports automatic RoPE scaling
    """int: Maximum sequence length for training and inference. Unsloth handles RoPE scaling automatically."""
    lora_r: int = 32
    """int: LoRA rank. Higher values increase model capacity but also trainable parameters."""
    lora_alpha: int = 64
    """int: LoRA alpha. Scaling factor, often set to 2 * lora_r."""
    lora_dropout: float = 0.0
    """float: Dropout probability for LoRA layers."""
    # --- Training Hyperparameters ---
    num_epochs: int = 6 # Reduced for demo speed, adjust as needed
    """int: Number of training epochs."""
    per_device_train_batch_size: int = 4
    """int: Batch size per device for training."""
    gradient_accumulation_steps: int = 16
    """int: Number of steps to accumulate gradients before an optimizer step."""
    learning_rate: float = 2e-4 # Common starting point for LoRA
    """float: Peak learning rate for the optimizer."""
    warmup_ratio: float = 0.1
    """float: Ratio of total steps to use for learning rate warmup."""
    weight_decay: float = 0.01
    """float: Weight decay coefficient for regularization."""
    max_grad_norm: float = 0.3 # Common for LoRA stability
    """float: Maximum gradient norm for clipping."""
    # --- SFTTrainer Specific ---
    packing: bool = False # Important for masked loss on responses
    """bool: Whether to pack sequences. False is recommended for response-only fine-tuning."""
    dataset_text_field: str = "text" # This is the field the SFTTrainer will use
    """str: The column name in the dataset containing the text to train on."""
    # --- Evaluation and Checkpointing ---
    eval_steps: int = 100
    """int: Number of update steps between evaluations."""
    save_steps: int = 100
    """int: Number of update steps between saving checkpoints."""
    logging_steps: int = 50
    """int: Number of update steps between logging."""
    save_total_limit: int = 100
    """int: Maximum number of checkpoints to keep."""
    # --- System Configuration ---
    seed: int = 42
    """int: Random seed for reproducibility."""
    fp16: bool = not is_bfloat16_supported() # Use fp16 if bfloat16 not available
    """bool: Enable mixed precision training with fp16."""
    bf16: bool = is_bfloat16_supported() # Use bfloat16 if supported (better for training)
    """bool: Enable mixed precision training with bfloat16 (preferred if available)."""
    # --- MLflow Configuration ---
    experiment_name: str = "qwen_unsloth_finetuning"
    """str: Name of the MLflow experiment."""
    run_name_prefix: str = "unsloth_token_filtered"
    """str: Prefix for the MLflow run name."""
    mlflow_tracking_uri: str = "./mlruns"
    """str: URI for the MLflow tracking server (local by default)."""
# ============================================================================
# End Configuration Section
# ============================================================================
# Language mappings for prompt templates
LANGUAGE_NAMES = {
    "en": "Anglais", "fr": "Français", "it": "Italien", 
    "es": "Espagnol", "de": "Allemand", "pt": "Portugais",
    "nl": "Néerlandais", "ru": "Russe", "zh-cn": "Chinois (Simplifié)",
    "ar": "Arabe", "ja": "Japonais", "ko": "Coréen", "hi": "Hindi"
}
# Task-specific prompt templates
TASK_PROMPTS = {
    "reco_gen": {
        "system": "Tu es Qwen Audit Reporter, créé par TMD-IGL. Tu es un assistant à la rédaction de rapport d'audit.",
        "template": "Analyse ce constat d'audit et génère des recommandations.\nCONSTAT:\n{instruction}\nGénère des recommandations SMART, précises et actionnables en {langue}."
    },
    "synth_gen_interm": {
        "system": "Tu es Qwen Audit Reporter, créé par TMD-IGL. Tu es un assistant à la rédaction de rapport d'audit.",
        "template": "Synthétise ces informations d'audit.\nINFORMATIONS:\n{instruction}\nProduis une synthèse intermédiaire structurée en {langue}."
    },
    "synth_gen_final": {
        "system": "Tu es Qwen Audit Reporter, créé par TMD-IGL. Tu es un assistant à la rédaction de rapport d'audit.",
        "template": "Synthétise ces informations d'audit.\nINFORMATIONS:\n{instruction}\nProduis une synthèse finale structurée en {langue}."
    },
    "constat_gen": {
        "system": "Tu es Qwen Audit Reporter, créé par TMD-IGL. Tu es un assistant à la rédaction de rapport d'audit.",
        "template": "Analyse ces informations et formule des constats.\nINFORMATIONS:\n{instruction}\nGénère des constats factuels en {langue}."
    }
}

class DataProcessor:
    """
    Handles data loading, filtering, and formatting for training.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None # Will be set later

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for data processing."""
        self.tokenizer = tokenizer

    def load_and_prepare_data(self) -> DatasetDict:
        """
        Loads data, filters by task and quality using the tokenizer, and formats it into a single text field
        suitable for the SFTTrainer.
        Returns:
            DatasetDict: A dictionary containing 'train', 'validation', and 'test' datasets.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before loading data.")
        logger.info("Using tokenizer for data formatting and filtering...")
        
        def format_sample(instruction: str, output: str, task: str = "reco_gen") -> str:
            """Formats a single instruction-output pair into chat template."""
            try:
                lang_code = detect(instruction[:1000]) # Increase sample size for detection
            except:
                lang_code = "fr"
            langue = LANGUAGE_NAMES.get(lang_code, "Français")
            task_config = TASK_PROMPTS.get(task, TASK_PROMPTS["reco_gen"])
            messages = [
                {"role": "system", "content": task_config["system"]},
                {"role": "user", "content": task_config["template"].format(instruction=instruction, langue=langue)},
                {"role": "assistant", "content": output}
            ]
            # Apply chat template to get the full formatted string
            # `add_generation_prompt` is False because we have the assistant's response
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return text

        logger.info(f"Loading datasets for tasks: {', '.join(self.config.selected_tasks)}")
        # Load raw data
        train_df = pd.read_json(self.config.train_path, lines=True)
        val_df = pd.read_json(self.config.val_path, lines=True)
        test_df = pd.read_json(self.config.test_path, lines=True)
        
        # Filter by selected tasks
        train_df = train_df[train_df["task"].isin(self.config.selected_tasks)].copy()
        val_df = val_df[val_df["task"].isin(self.config.selected_tasks)].copy()
        test_df = test_df[test_df["task"].isin(self.config.selected_tasks)].copy()
        
        # Basic cleaning (you can add more robust cleaning here if needed)
        train_df.dropna(subset=['instruction', 'output'], inplace=True)
        val_df.dropna(subset=['instruction', 'output'], inplace=True)
        test_df.dropna(subset=['instruction', 'output'], inplace=True)
        
        # --- Tokenizer-Based Filtering Step ---
        def filter_samples_with_tokenizer(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
            initial_count = len(df)
            logger.info(f"Starting tokenizer-based filtering for {split_name} set ({initial_count} samples)...")
            # 1. Format samples first to get the full text
            df['temp_formatted_text'] = df.apply(
                lambda row: format_sample(row['instruction'], row['output'], row['task']), axis=1
            )
            # 2. Tokenize the full formatted text to get total length and check structure
            logger.info(f"Tokenizing {split_name} samples for filtering...")
            # Tokenize in batch for efficiency
            encodings = self.tokenizer(
                df['temp_formatted_text'].tolist(),
                add_special_tokens=True,
                truncation=False, # Don't truncate yet, we want to check full length
                padding=False, # Don't pad
                return_attention_mask=False # We only need input_ids
            )
            df['temp_total_tokens'] = [len(ids) for ids in encodings['input_ids']]
            # 3. Tokenize just the instruction part to estimate prompt length
            df['temp_prompt_text'] = df.apply(
                lambda row: format_sample(row['instruction'], "", row['task']), axis=1
            )
            prompt_encodings = self.tokenizer(
                df['temp_prompt_text'].tolist(),
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False
            )
            df['temp_prompt_tokens'] = [len(ids) for ids in prompt_encodings['input_ids']]
            # 4. Calculate estimated response length
            df['temp_response_tokens'] = df['temp_total_tokens'] - df['temp_prompt_tokens']
            # 5. Apply filtering criteria based on token counts
            # a. Check total sequence length
            length_mask = df['temp_total_tokens'] <= self.config.max_seq_length
            # b. Check minimum response length (tokens)
            response_mask = df['temp_response_tokens'] >= self.config.min_assistant_response_tokens
            # Combine filters
            combined_mask = length_mask & response_mask
            # Filter the dataframe
            df_filtered = df[combined_mask].copy()
            final_count = len(df_filtered)
            removed_count = initial_count - final_count
            logger.info(f"Filtered {removed_count} samples from {split_name} set (Reasons: Length={initial_count - length_mask.sum()}, Response={initial_count - response_mask.sum()}). Final count: {final_count}")
            # Clean up temporary columns
            df_filtered.drop(columns=[
                'temp_formatted_text', 'temp_total_tokens', 
                'temp_prompt_text', 'temp_prompt_tokens', 'temp_response_tokens'
            ], inplace=True, errors='ignore')
            return df_filtered

        logger.info("Filtering potentially incomplete samples using tokenizer...")
        train_df = filter_samples_with_tokenizer(train_df, "training")
        val_df = filter_samples_with_tokenizer(val_df, "validation")
        test_df = filter_samples_with_tokenizer(test_df, "test")
        # --- End Tokenizer-Based Filtering ---
        
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
        
        # Format data for SFTTrainer (final formatting)
        logger.info("Formatting datasets for SFTTrainer...")
        train_df['text'] = train_df.apply(lambda row: format_sample(row['instruction'], row['output'], row['task']), axis=1)
        val_df['text'] = val_df.apply(lambda row: format_sample(row['instruction'], row['output'], row['task']), axis=1)
        test_df['text'] = test_df.apply(lambda row: format_sample(row['instruction'], row['output'], row['task']), axis=1)
        
        # Convert to HuggingFace datasets
        dataset_dict = DatasetDict({
            'train': HFDataset.from_pandas(train_df[['text']]),
            'validation': HFDataset.from_pandas(val_df[['text']]),
            'test': HFDataset.from_pandas(test_df[['text']])
        })
        
        return dataset_dict

class FineTuner:
    """
    Orchestrates the fine-tuning process using Unsloth and SFTTrainer.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model_and_tokenizer(self):
        """
        Loads the pre-trained model and tokenizer using Unsloth's optimized methods.
        Applies LoRA configuration.
        """
        logger.info("Setting up model and tokenizer with Unsloth...")
        # --- Load model and tokenizer with Unsloth ---
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype, # Auto-detects bfloat16/float16
            load_in_4bit=self.config.load_in_4bit, # Enables QLoRA
        )
        logger.info("✓ Model and tokenizer loaded with Unsloth.")
        
        # Ensure pad token is set correctly for training
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Pad token set to EOS token.")
            
        # --- Apply LoRA Adapters ---
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none", # Optimized setting
            use_gradient_checkpointing="unsloth", # Enable Unsloth's gradient checkpointing
            random_state=self.config.seed,
            use_rslora=False,
            loftq_config=None,
        )
        logger.info("✓ LoRA adapters applied.")
        # Print trainable parameters
        self.model.print_trainable_parameters()

    def train(self, dataset_dict: DatasetDict):
        """
        Configures and runs the training process using SFTTrainer.
        Args:
            dataset_dict (DatasetDict): The dictionary of datasets.
        """
        logger.info("Setting up training arguments and trainer...")
        # --- Configure Training Arguments ---
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            eval_strategy="steps", # Use 'steps' for evaluation during training
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            output_dir=self.config.output_dir,
            optim="adamw_8bit", # 8-bit Adam optimizer, efficient with QLoRA
            seed=self.config.seed,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            report_to=["mlflow"], # Enable MLflow logging
            run_name=f"{self.config.run_name_prefix}_{'_'.join(self.config.selected_tasks)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Disable grouping by length to simplify data handling with SFTTrainer
            group_by_length=False, 
            # Disable removal of unused columns to ensure 'text' field is available
            remove_unused_columns=False, 
        )
        
        # --- Initialize SFTTrainer ---
        # SFTTrainer handles tokenization, data collation, and crucially,
        # response-only loss calculation when packing=False.
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict['validation'],
            dataset_text_field=self.config.dataset_text_field, # Field containing formatted text
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing, # False for response-only loss
        )
        
        logger.info("✓ Trainer configured with response-only loss.")
        
        # --- Start MLflow Run ---
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        if mlflow.active_run():
            mlflow.end_run()
            
        with mlflow.start_run(run_name=training_args.run_name):
            # --- Log Parameters Explicitly (Avoiding Conflicts) ---
            PARAMS_TO_LOG = {
                "model_name": self.config.model_name,
                "selected_tasks": ','.join(self.config.selected_tasks),
                # --- Unsloth Parameters ---
                "load_in_4bit": self.config.load_in_4bit,
                "max_seq_length_intended": self.config.max_seq_length, # Log the intended value
                # --- LoRA Parameters ---
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                # --- Training Hyperparameters ---
                "num_epochs": self.config.num_epochs,
                "per_device_train_batch_size": self.config.per_device_train_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "effective_batch_size": self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps,
                "learning_rate": self.config.learning_rate,
                "warmup_ratio": self.config.warmup_ratio,
                "weight_decay": self.config.weight_decay,
                "max_grad_norm": self.config.max_grad_norm,
                # --- SFTTrainer Specific ---
                "packing": self.config.packing,
                # --- Data Filtering (Tokenizer-Based) ---
                "min_assistant_response_tokens": self.config.min_assistant_response_tokens,
                # --- System Configuration ---
                "seed": self.config.seed,
                "fp16": self.config.fp16,
                "bf16": self.config.bf16,
                # --- Dtype used (resolved by Unsloth) ---
                "dtype_used": str(self.config.dtype) if self.config.dtype else "Auto-detected (bfloat16/float16)"
            }
            
            # Log the explicitly selected parameters
            for key, value in PARAMS_TO_LOG.items():
                 try:
                     # MLflow usually handles str, int, float, bool
                     serializable_value = value
                     if isinstance(value, list):
                         serializable_value = ','.join(map(str, value)) # Convert lists to comma-separated string
                     elif not isinstance(value, (str, int, float, bool)):
                         serializable_value = str(value) # Convert other types to string
                     mlflow.log_param(key, serializable_value)
                     logger.debug(f"Logged parameter: {key} = {serializable_value}")
                 except Exception as e:
                     logger.warning(f"Could not log param {key} (Value: {serializable_value}): {e}")
                     
            # Log script as artifact
            script_path = __file__
            custom_script_name = "qwen_finetuner_unsloth"  # Custom name for the script
            mlflow.log_artifact(script_path, artifact_path=custom_script_name)
            
            # --- Train the Model ---
            logger.info("Starting training...")
            train_result = self.trainer.train()
            logger.info("Training completed.")
            
            # --- Save Model and Tokenizer ---
            logger.info("Saving model and tokenizer...")
            self.trainer.save_model() # Saves PEFT weights and config
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Save training config
            config_path = Path(self.config.output_dir) / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            mlflow.log_artifact(str(config_path))
            logger.info(f"Model, tokenizer, and config saved to {self.config.output_dir}")
            
            # --- Evaluate on Test Set ---
            logger.info("Evaluating on test set...")
            test_results = self.trainer.evaluate(eval_dataset=dataset_dict['test'], metric_key_prefix="test")
            # Log test results to MLflow
            for key, value in test_results.items():
                try:
                    mlflow.log_metric(key, value)
                except Exception as e:
                    logger.warning(f"Could not log test metric {key}: {e}")
            logger.info(f"Test Results: {test_results}")
            
            return train_result, test_results

def main():
    """
    Main execution function.
    """
    try:
        # Initialize configuration
        config = TrainingConfig()
        logger.info(f"Starting fine-tuning for tasks: {', '.join(config.selected_tasks)}")
        
        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            
        # --- Initialize Fine-Tuner ---
        fine_tuner = FineTuner(config)
        
        # --- Load and Prepare Data ---
        data_processor = DataProcessor(config)
        # Setup tokenizer for data formatting *before* loading data
        # Load a temporary tokenizer to format data
        logger.info("Loading temporary tokenizer for data processing...")
        temp_tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=False, # Don't need to load in 4bit for just tokenizing
        )[1] # Get tokenizer from the tuple returned by from_pretrained
        data_processor.set_tokenizer(temp_tokenizer)
        dataset_dict = data_processor.load_and_prepare_data()
        
        # --- Setup Model and Tokenizer (for training) ---
        fine_tuner.setup_model_and_tokenizer()
        
        # --- Train Model ---
        train_result, test_results = fine_tuner.train(dataset_dict)
        logger.info("Fine-tuning process completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Ensure MLflow run is ended
        if mlflow.active_run():
            mlflow.end_run()

if __name__ == "__main__":
    main()