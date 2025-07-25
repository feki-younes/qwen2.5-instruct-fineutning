"""
Audit Recommendation Generation with LLM Fine-tuning
Using Qwen2.5-1.5B-Instruct with LoRA for efficient training
Version 2.0 - Simplified for Windows with 48GB VRAM
"""

import os
# Disable transformers online mode
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*sparse_softmax_cross_entropy.*')
warnings.filterwarnings('ignore', module='transformers|datasets|evaluate')

# Essential imports
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import torch
from torch.nn import functional as F
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	TrainingArguments,
	Trainer,
	EarlyStoppingCallback,
	TrainerCallback,
	DataCollatorForLanguageModeling
)
from peft import (
	LoraConfig,
	get_peft_model,
	prepare_model_for_kbit_training,
	TaskType
)
from datasets import Dataset, DatasetDict, concatenate_datasets
import evaluate
import mlflow
import mlflow.pytorch
from tqdm.auto import tqdm
import gc
import logging
import sys
import platform
import io
import traceback
import json
from pathlib import Path
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure proper encoding for Windows
if platform.system() == 'Windows':
	os.system('chcp 65001 > nul 2>&1')
	sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
	sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s | %(levelname)s | %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S',
	handlers=[logging.StreamHandler(sys.stdout)],
	force=True
)

logger = logging.getLogger(__name__)

# Disable other loggers
for logger_name in ['tensorflow', 'transformers', 'datasets', 'evaluate']:
	logging.getLogger(logger_name).setLevel(logging.ERROR)


@dataclass
class TrainingConfig:
	"""Configuration class for LLM training parameters"""
	# Model settings
	model_path: str = "Qwen2.5-1.5B-Instruct"  # Fixed: Added Qwen/ prefix
	output_dir: str = "./recommendation_llm_finetuned"
	max_length: int = 2048  # Increased for 48GB VRAM
	
	# LoRA configuration - can be higher with 48GB VRAM
	lora_r: int = 128
	lora_alpha: int = 256
	lora_dropout: float = 0.1
	lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
	
	# Quantization settings
	use_4bit: bool = False  # With 48GB, we can use full precision
	bnb_4bit_compute_dtype: str = "float16"
	bnb_4bit_quant_type: str = "nf4"
	use_nested_quant: bool = True
	
	# Data paths
	train_data_path: str = "./data/train_data_w_aug.json"
	validation_data_path: str = "./data/validation_data_wo_aug.json"
	test_data_path: str = "./data/test_data_wo_aug.json"
	
	# Training hyperparameters - optimized for 48GB
	num_epochs: int = 3
	batch_size: int = 2  # Increased batch size
	gradient_accumulation_steps: int = 8  # Reduced accumulation
	learning_rate: float = 2e-4
	warmup_ratio: float = 0.03
	weight_decay: float = 0.001
	max_grad_norm: float = 0.3
	
	# Evaluation and saving
	eval_steps: int = 100
	save_steps: int = 100
	logging_steps: int = 25
	save_total_limit: int = 5
	save_on_each_eval: bool = True
	enable_generation_eval: bool = True  # Renamed from predict_with_generate
	
	# Early stopping
	early_stopping_patience: int = 3
	early_stopping_threshold: float = 0.001
	
	# Technical settings
	use_fp16: bool = True
	gradient_checkpointing: bool = True  # Not needed with 48GB
	chunk_size: int = 5000  # Larger chunks
	enable_full_training: bool = True
	local_files_only: bool = True
	seed: int = 42
	
	# Generation settings
	generation_max_new_tokens: int = 1024
	generation_temperature: float = 0.7  # Fixed: Was 0.7, should be 0.4 for consistency
	generation_top_p: float = 0.9
	generation_top_k: int = 50
	generation_repetition_penalty: float = 1.1
	
	# Monitoring
	gradient_history_size: int = 1000
	loss_history_size: int = 100
	
	# Chat template
	system_prompt: str = "Tu es un expert en audit et contrôle interne. Tu génères des recommandations professionnelles et détaillées basées sur les constats d'audit fournis."
	
	# Response markers
	assistant_start_token: str = "<|assistant|>"
	user_end_token: str = "<|user|>"
	
	# Data quality settings
	min_recommendation_words: int = 10
	max_recommendation_words: int = 900
	validate_augmented_data: bool = True
	augmentation_diversity_threshold: float = 0.7


class TableLogger:
	"""Utility class for formatted table logging"""
	
	@staticmethod
	def create_simple_table(title: str, data: Dict[str, Any]) -> str:
		lines = []
		lines.append("\n" + "="*100)
		lines.append(f" {title}")
		lines.append("="*100)
		
		if not data:
			lines.append(" No data available")
			return "\n".join(lines)
		
		max_key_len = min(max(len(str(k)) for k in data.keys()) + 2, 35)
		
		for key, value in data.items():
			if isinstance(value, float):
				if np.isnan(value):
					val_str = "NaN"
				elif np.isinf(value):
					val_str = "Inf"
				elif abs(value) >= 1000:
					val_str = f"{value:,.2f}"
				elif abs(value) >= 0.01:
					val_str = f"{value:.4f}"
				else:
					val_str = f"{value:.2e}"
			elif isinstance(value, int):
				val_str = f"{value:,}"
			else:
				val_str = str(value)
			
			if len(val_str) > 60:
				val_str = val_str[:60] + "..."
				
			lines.append(f" {str(key):<{max_key_len}} : {val_str}")
		
		lines.append("="*100)
		return "\n".join(lines)


def force_print(message):
	"""Force immediate output to console"""
	print(message, flush=True)
	sys.stdout.flush()


def safe_mlflow_log(log_func, key, value, max_retries: int = 3, **kwargs):
	"""Safely log to MLflow with retry logic"""
	for attempt in range(max_retries):
		try:
			log_func(key, value, **kwargs)
			return
		except Exception as e:
			if attempt == max_retries - 1:
				logger.warning(f"Failed to log to MLflow after {max_retries} attempts: {key} = {value}, Error: {e}")
			else:
				import time
				time.sleep(0.5 * (attempt + 1))


def set_seed(seed=42):
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


class MetricsComputer:
	"""Handles metric computation with proper alignment for generation"""
	
	def __init__(self, tokenizer, cache_dir: str = "./metrics"):
		self.tokenizer = tokenizer
		self.cache_dir = cache_dir
		self.bleu = None
		self.rouge = None
		self._initialize_metrics()
	
	def _initialize_metrics(self):
		"""Initialize metrics with multiple fallback strategies"""
		# Strategy 1: Try loading from cache
		try:
			self.bleu = evaluate.load("./metrics/bleu")
			self.rouge = evaluate.load("./metrics/rouge")
			force_print("✓ Metrics loaded from cache successfully")
			return
		except Exception as e:
			logger.warning(f"Failed to load metrics from cache: {e}")
		
		# Strategy 2: Try downloading (temporarily disable offline mode)
		try:
			logger.info("Attempting to download metrics...")
			os.environ.pop("HF_DATASETS_OFFLINE", None)
			os.environ.pop("TRANSFORMERS_OFFLINE", None)
			
			Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

			self.bleu = evaluate.load("bleu", cache_dir=self.cache_dir)
			self.rouge = evaluate.load("rouge", cache_dir=self.cache_dir)
			
			# Re-enable offline mode
			os.environ["HF_DATASETS_OFFLINE"] = "1"
			os.environ["TRANSFORMERS_OFFLINE"] = "1"
			
			force_print("✓ Metrics downloaded and cached successfully")
			return
		except Exception as e:
			logger.warning(f"Failed to download metrics: {e}")
			# Re-enable offline mode
			os.environ["HF_DATASETS_OFFLINE"] = "1"
			os.environ["TRANSFORMERS_OFFLINE"] = "1"
		
		# Strategy 3: Use simple implementations as fallback
		logger.warning("Using simplified metric implementations as fallback")
		self.bleu = None
		self.rouge = None
	
	def compute_structure_score(self, text: str) -> float:
		"""Score recommendation structure (paragraphs, bullets, etc.)"""
		lines = text.strip().split('\n')
		
		# Check for structured elements
		has_paragraphs = len([l for l in lines if len(l.strip()) > 50]) > 1
		has_bullets = any(l.strip().startswith(('-', '•', '*', '1.', '2.')) for l in lines)
		has_sections = any(':' in l and len(l) < 100 for l in lines)
		
		score = 0.0
		if has_paragraphs: score += 0.4
		if has_bullets: score += 0.3
		if has_sections: score += 0.3
		
		return score
	
	def compute_metrics_for_llm(self, eval_pred):
		"""Compute metrics for LLM with proper alignment"""
		# Handle different input formats
		if hasattr(eval_pred, 'predictions'):
			# New format: EvalPrediction object
			predictions = eval_pred.predictions
			labels = eval_pred.label_ids
		elif isinstance(eval_pred, tuple) and len(eval_pred) == 3:
			# Trainer with custom prediction_step returns (loss, predictions, labels)
			_, predictions, labels = eval_pred
		elif isinstance(eval_pred, tuple) and len(eval_pred) == 2:
			# Standard format: (predictions, labels)
			predictions, labels = eval_pred
		else:
			logger.error(f"Unsupported eval_pred format: {type(eval_pred)}")
			return {"error": "unsupported_format"}
		
		if predictions is None or labels is None:
			logger.error("Predictions or labels are None")
			return {"error": "null_inputs"}
		
		try:
			# Handle different prediction formats
			if isinstance(predictions, tuple):
				predictions = predictions[0]
			
			# Decode predictions and labels
			decoded_preds = []
			decoded_labels = []
			
			for pred, label in zip(predictions, labels):
				# Convert to numpy arrays if needed
				if isinstance(pred, torch.Tensor):
					pred = pred.cpu().numpy()
				if isinstance(label, torch.Tensor):
					label = label.cpu().numpy()
					
				# Remove invalid tokens (negative values and padding)
				pred = pred[(pred != self.tokenizer.pad_token_id) & (pred >= 0)]
				label = label[(label != -100) & (label != self.tokenizer.pad_token_id) & (label >= 0)]
				
				# Decode only if we have valid tokens
				if len(pred) > 0:
					pred_text = self.tokenizer.decode(pred, skip_special_tokens=True)
				else:
					pred_text = ""
					
				if len(label) > 0:
					label_text = self.tokenizer.decode(label, skip_special_tokens=True)
				else:
					label_text = ""
				
				# Clean up
				pred_text = pred_text.strip()
				label_text = label_text.strip()
				
				if pred_text and label_text:
					decoded_preds.append(pred_text)
					decoded_labels.append(label_text)				
			
			if not decoded_preds:
				logger.warning("No valid prediction/label pairs after filtering")
				return {"error": "no_valid_pairs"}
			
			result = {}
			
			# Compute BLEU
			if self.bleu is not None:
				try:
					bleu_result = self.bleu.compute(
						predictions=decoded_preds,
						references=[[label] for label in decoded_labels]
					)
					result["bleu"] = round(bleu_result["bleu"], 4)
					# Add individual BLEU scores
					for i in range(1, 5):
						if "precisions" in bleu_result and len(bleu_result["precisions"]) >= i:
							result[f"bleu_{i}"] = round(bleu_result["precisions"][i-1], 4)
				except Exception as e:
					logger.warning(f"BLEU computation failed: {e}")
					result["bleu"] = 0.0
			
			# Compute ROUGE
			if self.rouge is not None:
				try:
					rouge_result = self.rouge.compute(
						predictions=decoded_preds,
						references=decoded_labels,
						use_stemmer=True
					)
					for key in ["rouge1", "rouge2", "rougeL"]:
						if key in rouge_result:
							result[key] = round(rouge_result[key], 4)
				except Exception as e:
					logger.warning(f"ROUGE computation failed: {e}")
			
			# Compute structure score
			structure_scores = [self.compute_structure_score(pred) for pred in decoded_preds]
			
			# Add length metrics
			pred_lengths = [len(p.split()) for p in decoded_preds]
			label_lengths = [len(l.split()) for l in decoded_labels]
			
			result.update({
				"avg_pred_length": round(np.mean(pred_lengths), 2),
				"avg_label_length": round(np.mean(label_lengths), 2),
				"length_ratio": round(np.mean(pred_lengths) / max(np.mean(label_lengths), 1), 4),
				"num_samples": len(decoded_preds),
				"structure_score": round(np.mean(structure_scores), 4)
			})
			
			# Compute semantic similarity using TF-IDF
			try:
				vectorizer = TfidfVectorizer(max_features=1000)
				all_texts = decoded_preds + decoded_labels
				tfidf_matrix = vectorizer.fit_transform(all_texts)
				
				similarities = []
				for i in range(len(decoded_preds)):
					sim = cosine_similarity(
						tfidf_matrix[i:i+1], 
						tfidf_matrix[len(decoded_preds)+i:len(decoded_preds)+i+1]
					)[0][0]
					similarities.append(sim)
				
				result["semantic_similarity"] = round(np.mean(similarities), 4)
			except Exception as e:
				logger.warning(f"Semantic similarity computation failed: {e}")
			
			# Log sample for verification
			logger.info(f"Metrics computed successfully for {len(decoded_preds)} samples")
			if decoded_preds:
				logger.debug(f"Sample - Pred: '{decoded_preds[0][:100]}...'")
				logger.debug(f"Sample - Ref: '{decoded_labels[0][:100]}...'")
			
			return result
			
		except Exception as e:
			logger.error(f"Error in compute_metrics: {str(e)}")
			logger.error(f"Full traceback: {traceback.format_exc()}")
			return {"error": "computation_failed", "error_message": str(e)}


class GradientMonitoringCallback(TrainerCallback):
	"""Monitor gradients and loss patterns with bounded history"""
	
	def __init__(self, config: TrainingConfig):
		self.gradient_history = []
		self.loss_history = []
		self.stuck_loss_counter = 0
		self.max_gradient_history = config.gradient_history_size
		self.max_loss_history = config.loss_history_size
		
	def on_log(self, args, state, control, logs=None, **kwargs):
		if logs and state.is_world_process_zero:
			if "loss" in logs:
				current_loss = logs["loss"]
				self.loss_history.append(current_loss)
				
				if len(self.loss_history) > 10:
					recent_losses = self.loss_history[-10:]
					if np.std(recent_losses) < 1e-6:
						self.stuck_loss_counter += 1
						if self.stuck_loss_counter > 5:
							logger.warning(f"Loss appears stuck at {np.mean(recent_losses):.6f}")
					else:
						self.stuck_loss_counter = 0
				
				if len(self.loss_history) > self.max_loss_history:
					self.loss_history = self.loss_history[-self.max_loss_history:]
			
			if "grad_norm" in logs:
				grad_norm = logs["grad_norm"]
				self.gradient_history.append(grad_norm)
				
				if grad_norm < 1e-8:
					logger.warning(f"Very small gradient norm: {grad_norm:.2e}")
				elif grad_norm > 100:
					logger.warning(f"Large gradient norm: {grad_norm:.2f}")
				
				if len(self.gradient_history) > self.max_gradient_history:
					self.gradient_history = self.gradient_history[-self.max_gradient_history:]
				
				if state.global_step % 50 == 0 and len(self.gradient_history) > 10:
					recent_grads = self.gradient_history[-50:]
					grad_stats = {
						"Mean gradient norm": f"{np.mean(recent_grads):.4f}",
						"Std gradient norm": f"{np.std(recent_grads):.4f}",
						"Min gradient norm": f"{np.min(recent_grads):.4f}",
						"Max gradient norm": f"{np.max(recent_grads):.4f}"
					}
					force_print(TableLogger.create_simple_table("GRADIENT STATISTICS", grad_stats))


class MemoryCleanupCallback(TrainerCallback):
	"""AUDIT: Enhanced callback to proactively clean up GPU memory at critical points to free VRAM."""

	def _log_and_clear(self, stage: str, state):
		if not (state.is_world_process_zero and torch.cuda.is_available()):
			return

		force_print("\n" + "="*60)
		force_print(f"MEMORY MANAGEMENT: CLEARING VRAM BEFORE {stage.upper()}")
		force_print(f"Step: {state.global_step}")
		force_print("="*60)

		# Log memory before cleanup
		allocated_before = torch.cuda.memory_allocated() / 1024**3
		reserved_before = torch.cuda.memory_reserved() / 1024**3
		force_print(f"GPU memory BEFORE cleanup:")
		force_print(f"  - Allocated: {allocated_before:.2f} GB")
		force_print(f"  - Reserved:  {reserved_before:.2f} GB")

		# It's crucial to delete references to tensors that are no longer needed.
		# The Trainer manages the model and optimizer, so we don't delete them here.
		# We rely on Python's garbage collector to find unreferenced tensors.
		gc.collect()

		# Empty CUDA cache
		torch.cuda.empty_cache()
		torch.cuda.synchronize()

		# Log memory after cleanup
		allocated_after = torch.cuda.memory_allocated() / 1024**3
		reserved_after = torch.cuda.memory_reserved() / 1024**3
		freed_allocated = allocated_before - allocated_after

		force_print(f"GPU memory AFTER cleanup:")
		force_print(f"  - Allocated: {allocated_after:.2f} GB (Freed: {freed_allocated:.2f} GB)")
		force_print(f"  - Reserved:  {reserved_after:.2f} GB")
		force_print("="*60 + "\n")

	def on_evaluate(self, args: TrainingArguments, state , control, **kwargs):
		"""Called before each evaluation loop."""
		self._log_and_clear("EVALUATION", state)

	def on_train_end(self, args: TrainingArguments, state , control, **kwargs):
		"""Called at the end of the training process."""
		self._log_and_clear("FINAL TASKS (e.g., test evaluation)", state)


class GenerationQualityCallback(TrainerCallback):
	"""Monitor generation quality during evaluation"""
	
	def __init__(self, tokenizer, num_samples: int = 3):
		self.tokenizer = tokenizer
		self.num_samples = num_samples
		self.generation_history = []
		
	def on_evaluate(self, args, state, control, model=None, **kwargs):
		if state.is_world_process_zero and model is not None:
			eval_dataset = kwargs.get('eval_dataset')
			if eval_dataset and len(eval_dataset) > 0:
				sample_indices = np.random.choice(
					len(eval_dataset), 
					min(self.num_samples, len(eval_dataset)), 
					replace=False
				)
				
				force_print("\n" + "="*100)
				force_print(f"GENERATION QUALITY CHECK - Step {state.global_step}")
				force_print("="*100)
				
				for idx in sample_indices:
					sample = eval_dataset[int(idx)]
					
					# Generate prediction
					inputs = {k: torch.tensor(v).unsqueeze(0).to(model.device) 
							 for k, v in sample.items() if k in ['input_ids', 'attention_mask']}
					
					with torch.no_grad():
						generated = model.generate(
							**inputs,
							max_new_tokens=200,
							temperature=0.4,
							do_sample=True,
							pad_token_id=self.tokenizer.pad_token_id
						)
					
					generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
					
					# Extract recommendation part only
					if "Génère des recommandations" in generated_text:
						generated_text = generated_text.split("Génère des recommandations")[-1].strip()
					
					force_print(f"\nSample {idx + 1}:")
					force_print(f"Generated: {generated_text[:200]}...")
					
					self.generation_history.append({
						'step': state.global_step,
						'text': generated_text
					})
				
				force_print("="*100 + "\n")


class EnhancedMLflowCallback(TrainerCallback):
	"""Enhanced callback for MLflow and console logging"""
	
	def __init__(self):
		self.best_metrics = {
			'eval_loss': float('inf'),
			'eval_perplexity': float('inf'),
			'eval_bleu': 0,
			'eval_rougeL': 0,
			'eval_semantic_similarity': 0,
			'eval_structure_score': 0
		}
	
	def on_evaluate(self, args, state, control, metrics=None, **kwargs):
		if state.is_world_process_zero and metrics:
			# Log all metrics to MLflow
			for key, value in metrics.items():
				if isinstance(value, (int, float)):
					metric_name = key.replace("/", "_").replace(" ", "_")
					safe_mlflow_log(mlflow.log_metric, metric_name, value, step=state.global_step)
			
			# Calculate perplexity from loss
			eval_loss = metrics.get('eval_loss', float('inf'))
			eval_perplexity = np.exp(eval_loss) if eval_loss < 10 else float('inf')
			
			# Prepare display metrics
			display_metrics = {
				"Step": state.global_step,
				"Epoch": f"{state.epoch:.2f}" if state.epoch else "N/A",
				"Eval Loss": eval_loss,
				"Eval Perplexity": eval_perplexity
			}
			
			# Add all metric scores if available
			metric_mappings = {
				'eval_bleu': 'BLEU',
				'eval_bleu_1': 'BLEU-1',
				'eval_bleu_2': 'BLEU-2',
				'eval_bleu_3': 'BLEU-3',
				'eval_bleu_4': 'BLEU-4',
				'eval_rouge1': 'ROUGE-1',
				'eval_rouge2': 'ROUGE-2',
				'eval_rougeL': 'ROUGE-L',
				'eval_semantic_similarity': 'Semantic Sim',
				'eval_structure_score': 'Structure',
				'eval_runtime': 'Runtime (s)',
				'eval_samples_per_second': 'Samples/s'
			}
			
			for metric_key, display_name in metric_mappings.items():
				if metric_key in metrics:
					display_metrics[display_name] = metrics[metric_key]
			
			# Check for improvements
			improvements = []
			for metric, direction in [
				('eval_loss', 'down'),
				('eval_perplexity', 'down'),
				('eval_bleu', 'up'),
				('eval_rougeL', 'up'),
				('eval_semantic_similarity', 'up'),
				('eval_structure_score', 'up')
			]:
				if metric in metrics:
					current_value = metrics[metric] if metric != 'eval_perplexity' else eval_perplexity
					best_value = self.best_metrics.get(metric, float('inf') if direction == 'down' else 0)
					
					if direction == 'down' and current_value < best_value:
						self.best_metrics[metric] = current_value
						improvements.append(f"{metric.replace('eval_', '')} ↓")
					elif direction == 'up' and current_value > best_value:
						self.best_metrics[metric] = current_value
						improvements.append(f"{metric.replace('eval_', '')} ↑")
			
			if improvements:
				display_metrics["Improvements"] = ", ".join(improvements)
			
			# Add GPU metrics
			if torch.cuda.is_available():
				try:
					memory_allocated = torch.cuda.memory_allocated() / 1024**3
					memory_reserved = torch.cuda.memory_reserved() / 1024**3
					display_metrics["GPU Memory (GB)"] = f"{memory_allocated:.2f}/{memory_reserved:.2f}"
					display_metrics["GPU Utilization"] = f"{(memory_allocated / 48 * 100):.1f}%"
					
					safe_mlflow_log(mlflow.log_metric, "gpu_memory_allocated_gb", memory_allocated, step=state.global_step)
					safe_mlflow_log(mlflow.log_metric, "gpu_memory_reserved_gb", memory_reserved, step=state.global_step)
				except Exception as e:
					logger.debug(f"Failed to get GPU metrics: {e}")
			
			# Log metrics table
			force_print(TableLogger.create_simple_table(
				f"EVALUATION METRICS - Step {state.global_step}", 
				display_metrics
			))
			
			# Log best metrics summary
			best_summary = {
				"Best Loss": f"{self.best_metrics['eval_loss']:.4f}",
				"Best Perplexity": f"{self.best_metrics['eval_perplexity']:.4f}",
				"Best BLEU": f"{self.best_metrics['eval_bleu']:.4f}",
				"Best ROUGE-L": f"{self.best_metrics['eval_rougeL']:.4f}",
				"Best Semantic Sim": f"{self.best_metrics.get('eval_semantic_similarity', 0):.4f}",
				"Best Structure": f"{self.best_metrics.get('eval_structure_score', 0):.4f}"
			}
			force_print(TableLogger.create_simple_table("BEST METRICS SO FAR", best_summary))
	
	def on_log(self, args, state, control, logs=None, **kwargs):
		if state.is_world_process_zero and logs:
			for key, value in logs.items():
				if isinstance(value, (int, float)):
					metric_name = key.replace("/", "_").replace(" ", "_")
					safe_mlflow_log(mlflow.log_metric, metric_name, value, step=state.global_step)
			
			if state.global_step % args.eval_steps == 0 and state.global_step > 0:
				progress_info = {
					"Step": f"{state.global_step}/{state.max_steps}",
					"Progress": f"{(state.global_step / state.max_steps * 100):.1f}%",
					"Loss": f"{logs.get('loss', 'N/A'):.4f}" if 'loss' in logs else 'N/A',
					"Learning Rate": f"{logs.get('learning_rate', 0):.2e}",
					"Epoch": f"{state.epoch:.2f}" if state.epoch else "N/A"
				}
				
				if 'grad_norm' in logs:
					progress_info["Gradient Norm"] = f"{logs['grad_norm']:.4f}"
				
				force_print(TableLogger.create_simple_table("TRAINING PROGRESS", progress_info))

class CustomCheckpointCallback(TrainerCallback):
	"""Custom callback for model checkpointing"""
	
	def __init__(self, save_on_each_eval: bool = True):
		self.save_on_each_eval = save_on_each_eval
		
	def on_evaluate(self, args, state, control, model=None, **kwargs):
		if self.save_on_each_eval and state.is_world_process_zero:
			checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
			
			# Ensure directory exists
			checkpoint_dir.mkdir(parents=True, exist_ok=True)
			
			# Save model
			trainer = kwargs.get('trainer')
			if trainer:
				trainer.save_model(str(checkpoint_dir))
				trainer.save_state()
			
			# Save metrics
			metrics = kwargs.get('metrics', {})
			metrics_file = checkpoint_dir / "eval_metrics.json"
			try:
				with metrics_file.open('w', encoding='utf-8') as f:
					json.dump(metrics, f, indent=2, ensure_ascii=False)
			except OSError as e:
				logger.error(f"Failed to save metrics: {e}")
			
			# Log checkpoint info
			checkpoint_info = {
				"Step": state.global_step,
				"Path": str(checkpoint_dir),
				"Eval Loss": metrics.get('eval_loss', 'N/A'),
				"BLEU": metrics.get('eval_bleu', 'N/A'),
				"ROUGE-L": metrics.get('eval_rougeL', 'N/A'),
				"Semantic Sim": metrics.get('eval_semantic_similarity', 'N/A')
			}
			force_print(TableLogger.create_simple_table("CHECKPOINT SAVED", checkpoint_info))

class AuditRecommendationDataProcessor:
	"""Processes audit data for LLM training with validation"""
	
	def __init__(self, config: TrainingConfig):
		self.config = config
	
	def get_column_stats(self, df, column_name):
		lengths = df[column_name].apply(lambda x: len(str(x).split()))
		return {
			"Mean": f"{lengths.mean():.1f}",
			"Max": f"{lengths.max()}",
			"95th percentile": f"{lengths.quantile(0.95):.1f}",
			"99th percentile": f"{lengths.quantile(0.99):.1f}"
		}
	
	def validate_augmented_data(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, float]:
		"""Validate augmented data quality"""
		metrics = {}
		
		# Check for duplicates
		duplicates = df.duplicated(subset=['constat', 'recommandations']).sum()
		metrics['duplicate_rate'] = duplicates / len(df)
		
		# Check diversity using TF-IDF
		try:
			vectorizer = TfidfVectorizer(max_features=1000)
			
			# Sample for efficiency
			sample_size = min(1000, len(df))
			sample_df = df.sample(n=sample_size, random_state=42)
			
			# Compute diversity for constats
			constat_vectors = vectorizer.fit_transform(sample_df['constat'])
			constat_similarity = cosine_similarity(constat_vectors)
			np.fill_diagonal(constat_similarity, 0)  # Ignore self-similarity
			metrics['constat_avg_similarity'] = constat_similarity.mean()
			
			# Compute diversity for recommandations
			recomm_vectors = vectorizer.fit_transform(sample_df['recommandations'])
			recomm_similarity = cosine_similarity(recomm_vectors)
			np.fill_diagonal(recomm_similarity, 0)
			metrics['recomm_avg_similarity'] = recomm_similarity.mean()
			
		except Exception as e:
			logger.warning(f"Failed to compute diversity metrics: {e}")
		
		# Length statistics
		constat_lengths = df['constat'].apply(lambda x: len(str(x).split()))
		recomm_lengths = df['recommandations'].apply(lambda x: len(str(x).split()))
		
		metrics.update({
			'constat_avg_length': constat_lengths.mean(),
			'recomm_avg_length': recomm_lengths.mean(),
			'short_constats': (constat_lengths < 10).sum() / len(df),
			'short_recomms': (recomm_lengths < self.config.min_recommendation_words).sum() / len(df),
			'long_recomms': (recomm_lengths > self.config.max_recommendation_words).sum() / len(df)
		})
		
		# Log validation results
		validation_info = {
			f"{dataset_name} - Duplicates": f"{metrics['duplicate_rate']:.2%}",
			f"{dataset_name} - Constat similarity": f"{metrics.get('constat_avg_similarity', 0):.3f}",
			f"{dataset_name} - Recomm similarity": f"{metrics.get('recomm_avg_similarity', 0):.3f}",
			f"{dataset_name} - Short constats": f"{metrics['short_constats']:.2%}",
			f"{dataset_name} - Short recomms": f"{metrics['short_recomms']:.2%}",
			f"{dataset_name} - Long recomms": f"{metrics['long_recomms']:.2%}"
		}
		
		force_print(TableLogger.create_simple_table(f"{dataset_name.upper()} VALIDATION", validation_info))
		
		return metrics
	
	def load_data(self):
		force_print("\nLoading datasets...")
		
		# Load data
		train_df = pd.read_json(self.config.train_data_path)
		val_df = pd.read_json(self.config.validation_data_path)
		test_df = pd.read_json(self.config.test_data_path)
		
		# Rename columns
		for df in [train_df, val_df, test_df]:
			df.rename(columns={
				'instruction': 'constat',
				'output': 'recommandations'
			}, inplace=True)
		
		# Shuffle and store
		self.train_df = train_df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
		self.val_df = val_df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
		self.test_df = test_df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
		
		# Validate augmented data if enabled
		if self.config.validate_augmented_data:
			train_metrics = self.validate_augmented_data(self.train_df, "Train")
			val_metrics = self.validate_augmented_data(self.val_df, "Validation")
			test_metrics = self.validate_augmented_data(self.test_df, "Test")
			
			# Log to MLflow
			for name, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
				for key, value in metrics.items():
					safe_mlflow_log(mlflow.log_param, f"data_quality/{name}_{key}", value)
		
		# Filter out low-quality samples
		original_train_size = len(self.train_df)
		self.train_df = self.train_df[
			(self.train_df['constat'].str.split().str.len() >= 10) &
			(self.train_df['recommandations'].str.split().str.len() >= self.config.min_recommendation_words) &
			(self.train_df['recommandations'].str.split().str.len() <= self.config.max_recommendation_words)
		]
		filtered_count = original_train_size - len(self.train_df)
		
		if filtered_count > 0:
			force_print(f"Filtered {filtered_count} low-quality training samples")
		
		# Create dataset dict
		self.dataset_dict = DatasetDict({
			'train': Dataset.from_pandas(self.train_df),
			'validation': Dataset.from_pandas(self.val_df),
			'test': Dataset.from_pandas(self.test_df)
		})
		
		data_info = {
			"Train Samples": len(self.dataset_dict['train']),
			"Validation Samples": len(self.dataset_dict['validation']),
			"Test Samples": len(self.dataset_dict['test']),
			"Total Samples": sum(len(self.dataset_dict[split]) for split in ['train', 'validation', 'test']),
			"Filtered Samples": filtered_count
		}
		
		force_print(TableLogger.create_simple_table("DATASET STATISTICS", data_info))
		
		for key, value in data_info.items():
			safe_mlflow_log(mlflow.log_param, f"data/{key.lower().replace(' ', '_')}", value)
		
		# Word count statistics
		stats_info = {}
		for name, df in [("Train", self.train_df), ("Val", self.val_df), ("Test", self.test_df)]:
			constat_stats = self.get_column_stats(df, "constat")
			recomm_stats = self.get_column_stats(df, "recommandations")
			
			stats_info[f"{name} - Constat words"] = f"Mean: {constat_stats['Mean']} | Max: {constat_stats['Max']}"
			stats_info[f"{name} - Recomm words"] = f"Mean: {recomm_stats['Mean']} | Max: {recomm_stats['Max']}"
		
		force_print(TableLogger.create_simple_table("WORD COUNT STATISTICS", stats_info))
		
		return self.dataset_dict


class CustomDataCollatorForCausalLM(DataCollatorForLanguageModeling):
	"""Custom data collator that properly masks prompt tokens"""
	
	def __init__(self, tokenizer, mlm=False, pad_to_multiple_of=None, 
				 prompt_end_marker="Génère des recommandations détaillées pour ce constat."):
		super().__init__(tokenizer=tokenizer, mlm=mlm, pad_to_multiple_of=pad_to_multiple_of)
		self.prompt_end_marker = prompt_end_marker
		self.prompt_end_tokens = tokenizer.encode(prompt_end_marker, add_special_tokens=False)
	
	def __call__(self, features):
		# First, pad the features
		batch = self.tokenizer.pad(
			features,
			padding=True,
			max_length=None,  # Pad to longest in batch
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt"
		)
		
		# Create labels from input_ids
		batch["labels"] = batch["input_ids"].clone()
		
		# Now mask the prompt portions
		for i in range(len(batch["input_ids"])):
			input_list = batch["input_ids"][i].tolist()
			
			# Find where the prompt ends
			prompt_end_pos = -1
			for j in range(len(input_list) - len(self.prompt_end_tokens) + 1):
				if input_list[j:j+len(self.prompt_end_tokens)] == self.prompt_end_tokens:
					prompt_end_pos = j + len(self.prompt_end_tokens)
					break
			
			# Mask everything before the assistant response
			if prompt_end_pos > 0:
				batch["labels"][i, :prompt_end_pos] = -100
			
			# Also mask padding tokens
			if self.tokenizer.pad_token_id is not None:
				batch["labels"][i, batch["input_ids"][i] == self.tokenizer.pad_token_id] = -100
		
		return batch

class CustomTrainerWithGeneration(Trainer):
	def __init__(self, *args, generation_config=None, enable_generation_eval=True, processor=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.generation_config = generation_config or {}
		self.enable_generation_eval = enable_generation_eval
		self.processor = processor

	def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
		if not self.enable_generation_eval or prediction_loss_only:
			return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

		with torch.no_grad():
			loss = None
			with self.compute_loss_context_manager():
				loss, _ = self.compute_loss(model, inputs, return_outputs=True)
			loss = loss.mean().detach()

		model.eval()
		input_ids = inputs["input_ids"]
		labels = inputs["labels"]
		batch_size = input_ids.shape[0]
		max_prompt_length = 0

		for i in range(batch_size):
			non_masked = (labels[i] != -100).nonzero(as_tuple=True)[0]
			if len(non_masked) > 0:
				prompt_length = non_masked[0].item()
				max_prompt_length = max(max_prompt_length, prompt_length)

		generation_inputs = {
			"input_ids": input_ids[:, :max_prompt_length] if max_prompt_length > 0 else input_ids,
			"attention_mask": inputs["attention_mask"][:, :max_prompt_length] if max_prompt_length > 0 else inputs["attention_mask"]
		}

		pad_token_id = self.processor.pad_token_id if hasattr(self.processor, "pad_token_id") else None
		eos_token_id = self.processor.eos_token_id if hasattr(self.processor, "eos_token_id") else None

		with torch.no_grad():
			generated_tokens = model.generate(
				**generation_inputs,
				max_new_tokens=self.generation_config.get("max_new_tokens", 768),
				temperature=self.generation_config.get("temperature", 0.4),
				top_p=self.generation_config.get("top_p", 0.9),
				top_k=self.generation_config.get("top_k", 50),
				repetition_penalty=self.generation_config.get("repetition_penalty", 1.1),
				pad_token_id=pad_token_id,
				eos_token_id=eos_token_id,
				do_sample=True
			)

		if max_prompt_length > 0:
			generated_portion = generated_tokens[:, max_prompt_length:]
		else:
			generated_portion = generated_tokens[:, input_ids.shape[1]:]

		label_portion = []
		for i in range(batch_size):
			non_masked_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
			if len(non_masked_indices) > 0:
				label_portion.append(labels[i][non_masked_indices])
			else:
				label_portion.append(torch.tensor([pad_token_id], device=labels.device))

		max_label_length = max(len(l) for l in label_portion)
		padded_labels = torch.full(
			(batch_size, max_label_length), 
			self.processor.pad_token_id,  # Use pad_token_id, not -100
			dtype=labels.dtype,
			device=labels.device
		)

		for i, l in enumerate(label_portion):
			padded_labels[i, :len(l)] = l

		return (loss, generated_portion, padded_labels)


class LLMRecommendationGenerator:
	"""Main class for training and using the LLM"""
	
	def __init__(self, config: TrainingConfig):
		self.config = config
		
		model_config = {
			"Model": config.model_path,
			"Max length": config.max_length,
			"LoRA rank": config.lora_r,
			"LoRA alpha": config.lora_alpha,
			"4-bit quantization": config.use_4bit,
			"Gradient checkpointing": config.gradient_checkpointing,
			"System": "Windows",
			"VRAM": "48GB"
		}
		force_print(TableLogger.create_simple_table("MODEL CONFIGURATION", model_config))
		
		force_print(f"\nLoading model from: {config.model_path}")
		
		try:
			# Load tokenizer
			self.tokenizer = AutoTokenizer.from_pretrained(
				config.model_path,
				trust_remote_code=True,
				local_files_only=config.local_files_only,
				padding_side="left"  # Add this
			)
			
			# Set padding token if not set
			if self.tokenizer.pad_token is None:
				self.tokenizer.pad_token = self.tokenizer.eos_token
				self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
			
			# Quantization config for 4-bit (optional with 48GB)
			if config.use_4bit:
				compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
				bnb_config = BitsAndBytesConfig(
					load_in_4bit=True,
					bnb_4bit_compute_dtype=compute_dtype,
					bnb_4bit_quant_type=config.bnb_4bit_quant_type,
					bnb_4bit_use_double_quant=config.use_nested_quant
				)
			else:
				bnb_config = None
			
			# Load model
			self.model = AutoModelForCausalLM.from_pretrained(
				config.model_path,
				quantization_config=bnb_config,
				device_map="auto",
				trust_remote_code=True,
				local_files_only=config.local_files_only,
				torch_dtype=torch.float16 if config.use_fp16 else torch.float32
			)
			
			# Prepare model for k-bit training if using quantization
			if config.use_4bit:
				self.model = prepare_model_for_kbit_training(self.model)
			
			# Configure LoRA
			lora_config = LoraConfig(
				r=config.lora_r,
				lora_alpha=config.lora_alpha,
				target_modules=config.lora_target_modules,
				lora_dropout=config.lora_dropout,
				bias="none",
				task_type=TaskType.CAUSAL_LM
			)
			
			# Apply LoRA
			self.model = get_peft_model(self.model, lora_config)
			self.model.print_trainable_parameters()
			
			# Enable gradient checkpointing if needed
			if config.gradient_checkpointing:
				self.model.gradient_checkpointing_enable()
				self.model.enable_input_require_grads()
			
			self.device = next(self.model.parameters()).device
			
		except Exception as e:
			logger.error(f"Failed to load model: {e}")
			raise
		
		system_info = {
			"Device": str(self.device),
			"PyTorch version": torch.__version__,
			"CUDA available": torch.cuda.is_available(),
			"Windows version": platform.version()
		}
		
		if torch.cuda.is_available():
			gpu_props = torch.cuda.get_device_properties(0)
			system_info.update({
				"GPU name": gpu_props.name,
				"GPU memory (GB)": f"{gpu_props.total_memory / 1024**3:.1f}",
			})
		
		force_print(TableLogger.create_simple_table("SYSTEM INFORMATION", system_info))
		
		# Initialize metrics computer
		self.metrics_computer = MetricsComputer(self.tokenizer)
	
	def format_prompt(self, constat: str, recommandation: Optional[str] = None) -> str:
		"""Format prompt using chat template"""
		messages = [
			{"role": "system", "content": self.config.system_prompt},
			{"role": "user", "content": f"Constat d'audit:\n{constat}\n\nGénère des recommandations détaillées pour ce constat."}
		]
		
		if recommandation:
			messages.append({"role": "assistant", "content": recommandation})
		
		return self.tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=recommandation is None
		)
		
	def preprocess_function(self, examples):
		"""Tokenize inputs for training - let collator handle labels"""
		texts = []
		for constat, recommandation in zip(examples["constat"], examples["recommandations"]):
			# Skip if too short
			if len(constat.split()) < 10 or len(recommandation.split()) < 5:
				continue
			
			text = self.format_prompt(constat, recommandation)
			texts.append(text)
		
		if not texts:
			return {
				"input_ids": [],
				"attention_mask": []
			}
		
		# Tokenize without padding and WITHOUT creating labels
		model_inputs = self.tokenizer(
			texts,
			max_length=self.config.max_length,
			truncation=True,
			padding=False,  # No padding
			return_tensors=None
		)
		
		# Don't create labels here - let the collator do it
		return model_inputs
	
	
	def tokenize_in_chunks(self, dataset, chunk_size=None):
		"""Process dataset in chunks with progress tracking"""
		if chunk_size is None:
			chunk_size = self.config.chunk_size
		
		tokenized_chunks = []
		
		if self.config.enable_full_training:
			total_items = len(dataset)
			desc = "Tokenizing (full dataset)"
		else:
			total_items = min(chunk_size * 2, len(dataset))
			desc = "Tokenizing (test mode)"
			logger.warning(f"Test mode: processing only {total_items} items")
		
		for i in tqdm(range(0, total_items, chunk_size), desc=desc):
			chunk_end = min(i + chunk_size, total_items)
			chunk = dataset.select(range(i, chunk_end))
			
			tokenized_chunk = chunk.map(
				self.preprocess_function,
				batched=True,
				remove_columns=chunk.column_names,
				desc=f"Processing chunk {i//chunk_size + 1}"
			)
			
			# Filter out empty samples
			tokenized_chunk = tokenized_chunk.filter(
				lambda x: len(x['input_ids']) > 0
			)
			
			if len(tokenized_chunk) > 0:
				tokenized_chunks.append(tokenized_chunk)
			
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		
		return concatenate_datasets(tokenized_chunks) if tokenized_chunks else dataset.select([])
	
	def train(self, dataset_dict: DatasetDict, resume_from_checkpoint: Optional[str] = None):
		"""Train the LLM with LoRA"""
		
		# End any existing active run
		try:
			mlflow.end_run()
		except Exception:
			pass
		
		experiment_name = f"audit_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
		mlflow.set_experiment(experiment_name)
		
		run_name = f"qwen_lr{self.config.learning_rate}_r{self.config.lora_r}_{datetime.now().strftime('%H%M%S')}"
		
		with mlflow.start_run(run_name=run_name) as run:
			force_print(f"\nMLflow run started: {run.info.run_id}")
			
			training_config = {
				"model": self.config.model_path,
				"output_dir": self.config.output_dir,
				"num_epochs": self.config.num_epochs,
				"batch_size": self.config.batch_size,
				"effective_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps,
				"learning_rate": self.config.learning_rate,
				"warmup_ratio": self.config.warmup_ratio,
				"lora_r": self.config.lora_r,
				"lora_alpha": self.config.lora_alpha,
				"algorithm": "LoRA Fine-tuning",
				"quantization": "4-bit" if self.config.use_4bit else "none",
				"generation_temperature": self.config.generation_temperature,
				"max_length": self.config.max_length,
				"system": "Windows 48GB VRAM"
			}
			
			force_print(TableLogger.create_simple_table("TRAINING CONFIGURATION", training_config))
			'''
			for key, value in training_config.items():
				safe_mlflow_log(mlflow.log_param, key, value)
			'''
			try:
				force_print("\nTokenizing datasets...")
				tokenized_datasets = DatasetDict({
					"train": self.tokenize_in_chunks(dataset_dict["train"]),
					"validation": self.tokenize_in_chunks(dataset_dict["validation"]),
					"test": self.tokenize_in_chunks(dataset_dict["test"])
				})
				
				tokenized_info = {
					"Train samples": len(tokenized_datasets['train']),
					"Validation samples": len(tokenized_datasets['validation']),
					"Test samples": len(tokenized_datasets['test'])
				}
				force_print(TableLogger.create_simple_table("TOKENIZED DATASET SIZES", tokenized_info))
				
				# Custom data collator with proper label masking
				data_collator = CustomDataCollatorForCausalLM(
					tokenizer=self.tokenizer,
					mlm=False,
					pad_to_multiple_of=8,
					prompt_end_marker="Génère des recommandations détaillées pour ce constat."
				)
				
				# Training arguments optimized for 48GB VRAM
				training_args = TrainingArguments(
					output_dir=self.config.output_dir,
					eval_strategy="steps",
					eval_steps=self.config.eval_steps,
					logging_strategy="steps",
					logging_steps=self.config.logging_steps,
					save_strategy="steps",
					save_steps=self.config.save_steps,
					save_total_limit=self.config.save_total_limit,
					learning_rate=self.config.learning_rate,
					per_device_train_batch_size=self.config.batch_size,
					per_device_eval_batch_size=4, #self.config.batch_size // 2,  # Conservative for eval
					gradient_accumulation_steps=self.config.gradient_accumulation_steps,
					warmup_ratio=self.config.warmup_ratio,
					num_train_epochs=self.config.num_epochs,
					weight_decay=self.config.weight_decay,
					load_best_model_at_end=True,
					metric_for_best_model="eval_loss",
					greater_is_better=False,
					fp16=self.config.use_fp16 and torch.cuda.is_available(),
					bf16=False,
					optim="adamw_torch" if not self.config.use_4bit else "paged_adamw_32bit",
					gradient_checkpointing=self.config.gradient_checkpointing,
					gradient_checkpointing_kwargs={"use_reentrant": False} if self.config.gradient_checkpointing else None,
					max_grad_norm=self.config.max_grad_norm,
					report_to=["mlflow"],
					logging_dir=f"{self.config.output_dir}/logs",
					seed=self.config.seed,
					dataloader_num_workers=0,  # For Windows
					remove_unused_columns=False,
					dataloader_pin_memory=True,
					resume_from_checkpoint=resume_from_checkpoint,
					disable_tqdm=False,
					#log_level="info",
					logging_nan_inf_filter=True,
					include_inputs_for_metrics=True  # Important for generation
				)
				
				# Generation config for evaluation
				generation_config = {
					"max_new_tokens": self.config.generation_max_new_tokens,
					"temperature": self.config.generation_temperature,
					"top_p": self.config.generation_top_p,
					"top_k": self.config.generation_top_k,
					"repetition_penalty": self.config.generation_repetition_penalty
				}
				
				force_print("\nInitializing trainer...")
				trainer = CustomTrainerWithGeneration(
					model=self.model,
					args=training_args,
					train_dataset=tokenized_datasets["train"],
					eval_dataset=tokenized_datasets["validation"],
					data_collator=data_collator,
					compute_metrics=self.metrics_computer.compute_metrics_for_llm if self.config.enable_generation_eval else None,
					generation_config=generation_config,
					enable_generation_eval=self.config.enable_generation_eval,
					processor=self.tokenizer
				)
				
				# Add custom callbacks
				trainer.add_callback(GradientMonitoringCallback(self.config))
				trainer.add_callback(MemoryCleanupCallback())
				trainer.add_callback(EnhancedMLflowCallback())
				trainer.add_callback(GenerationQualityCallback(self.tokenizer, num_samples=2))
				trainer.add_callback(CustomCheckpointCallback(self.config.save_on_each_eval))
				trainer.add_callback(EarlyStoppingCallback(
					early_stopping_patience=self.config.early_stopping_patience,
					early_stopping_threshold=self.config.early_stopping_threshold
				))
				
				training_start_info = {
					"Total training steps": trainer.args.max_steps,
					"Steps per epoch": len(tokenized_datasets["train"]) // (self.config.batch_size * self.config.gradient_accumulation_steps),
					"Start time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
					"Estimated GPU usage": f"~{(self.config.batch_size * 2048 * 2) / 1024:.1f}GB"
				}
				force_print(TableLogger.create_simple_table("TRAINING STARTED", training_start_info))
				
				# Train
				train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
				
				# Save model
				trainer.save_model()
				self.tokenizer.save_pretrained(self.config.output_dir)
				force_print(f"\nModel saved to: {self.config.output_dir}")
				
				# Evaluate on test set with generation
				force_print("\nEvaluating on test set...")
				test_results = trainer.evaluate(
					tokenized_datasets["test"], 
					metric_key_prefix="test"
				)
				
				# Prepare test metrics display
				test_metrics = {
					"Test Loss": test_results.get("test_loss", "N/A"),
					"Test Perplexity": np.exp(test_results["test_loss"]) if test_results.get("test_loss", 10) < 10 else "N/A"
				}
				
				# Add all available metrics
				metric_mappings = {
					"test_bleu": "BLEU",
					"test_bleu_1": "BLEU-1",
					"test_bleu_2": "BLEU-2", 
					"test_bleu_3": "BLEU-3",
					"test_bleu_4": "BLEU-4",
					"test_rouge1": "ROUGE-1",
					"test_rouge2": "ROUGE-2",
					"test_rougeL": "ROUGE-L",
					"test_semantic_similarity": "Semantic Sim",
					"test_structure_score": "Structure"
				}
				
				for metric_key, display_name in metric_mappings.items():
					if metric_key in test_results:
						test_metrics[display_name] = test_results[metric_key]
				
				test_metrics.update({
					"Runtime (s)": test_results.get("test_runtime", "N/A"),
					"Samples/s": test_results.get("test_samples_per_second", "N/A")
				})
				
				force_print(TableLogger.create_simple_table("TEST SET EVALUATION", test_metrics))
				
				# Log final metrics to MLflow
				final_metrics = {
					"final_train_loss": train_result.training_loss,
					"final_test_loss": test_results.get("test_loss", 0),
					"final_test_perplexity": np.exp(test_results.get("test_loss", 0)) if test_results.get("test_loss", 0) < 10 else 0
				}
				
				# Add all final test metrics
				for metric in ["bleu", "rouge1", "rouge2", "rougeL", "semantic_similarity", "structure_score"]:
					key = f"test_{metric}"
					if key in test_results:
						final_metrics[f"final_{key}"] = test_results[key]
				
				for key, value in final_metrics.items():
					safe_mlflow_log(mlflow.log_metric, key, value)
				
				force_print(TableLogger.create_simple_table("FINAL METRICS", final_metrics))
				
				# Save training configuration
				config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
				mlflow.log_dict(config_dict, "training_config.json")
				
				# Save LoRA config
				lora_config_dict = {
					"r": self.config.lora_r,
					"lora_alpha": self.config.lora_alpha,
					"lora_dropout": self.config.lora_dropout,
					"target_modules": self.config.lora_target_modules
				}
				mlflow.log_dict(lora_config_dict, "lora_config.json")
				
				# Save checkpoints list
				checkpoints = [d for d in os.listdir(self.config.output_dir) if d.startswith("checkpoint-")]
				if checkpoints:
					checkpoint_info = {
						"Total checkpoints": len(checkpoints),
						"Checkpoints": ", ".join(sorted(checkpoints)[:5]) + ("..." if len(checkpoints) > 5 else "")
					}
					force_print(TableLogger.create_simple_table("SAVED CHECKPOINTS", checkpoint_info))
				
			except Exception as e:
				logger.error(f"Training failed: {e}")
				logger.error(f"Full traceback: {traceback.format_exc()}")
				raise
			finally:
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
		
		completion_info = {
			"Status": "Training completed successfully",
			"Model saved to": self.config.output_dir,
			"MLflow run ID": run.info.run_id,
			"End time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		}
		force_print(TableLogger.create_simple_table("TRAINING SUMMARY", completion_info))
		
		return trainer
	
	def generate_recommendations(self, constat: str, validate: bool = True) -> str:
		"""Generate recommendations using the fine-tuned LLM with validation"""
		self.model.eval()
		
		prompt = self.format_prompt(constat)
		
		inputs = self.tokenizer(
			prompt,
			return_tensors="pt",
			truncation=True,
			max_length=self.config.max_length
		).to(self.device)
		
		with torch.no_grad():
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=self.config.generation_max_new_tokens,
				temperature=self.config.generation_temperature,
				top_p=self.config.generation_top_p,
				top_k=self.config.generation_top_k,
				repetition_penalty=self.config.generation_repetition_penalty,
				pad_token_id=self.tokenizer.pad_token_id,
				eos_token_id=self.tokenizer.eos_token_id,
				do_sample=True
			)
		
		response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
		
		# Extract only the assistant's response
		if "Génère des recommandations détaillées pour ce constat." in response:
			response = response.split("Génère des recommandations détaillées pour ce constat.")[-1].strip()
		
		# Validate response if enabled
		if validate:
			word_count = len(response.split())
			if word_count < self.config.min_recommendation_words:
				logger.warning(f"Generated recommendation too short: {word_count} words")
			elif word_count > self.config.max_recommendation_words:
				logger.warning(f"Generated recommendation too long: {word_count} words")
				# Truncate to max length
				words = response.split()[:self.config.max_recommendation_words]
				response = " ".join(words) + "..."
		
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		return response


def parse_args():
	"""Parse command line arguments"""
	parser = argparse.ArgumentParser(description="Fine-tune Qwen 1.5B for audit recommendations")
	
	parser.add_argument("--model-name", type=str, default="Qwen2.5-1.5B-Instruct",
					  help="Model name from HuggingFace")
	parser.add_argument("--output-dir", type=str, default="./recommendation_llm_finetuned",
					  help="Output directory")
	parser.add_argument("--num-epochs", type=int, default=3,
					  help="Number of training epochs")
	parser.add_argument("--batch-size", type=int, default=2,
					  help="Training batch size")
	parser.add_argument("--learning-rate", type=float, default=2e-4,
					  help="Learning rate")
	parser.add_argument("--lora-r", type=int, default=128,
					  help="LoRA rank")
	parser.add_argument("--use-4bit", action="store_true",
					  help="Enable 4-bit quantization (not needed with 48GB)")
	parser.add_argument("--test-mode", action="store_true",
					  help="Run in test mode with limited data")
	parser.add_argument("--local-only", action="store_true",
					  help="Use only local files")
	parser.add_argument("--resume-from", type=str, default=None,
					  help="Resume from checkpoint")
	parser.add_argument("--temperature", type=float, default=0.7,
					  help="Generation temperature")
	parser.add_argument("--no-validation", action="store_true",
					  help="Disable data validation")
	parser.add_argument("--max-length", type=int, default=2048,
					  help="Maximum sequence length")
	
	return parser.parse_args()


def main():
	"""Main training pipeline"""
	args = parse_args()
	
	# Create configuration
	config = TrainingConfig(
		model_path=args.model_name,
		output_dir=args.output_dir,
		num_epochs=args.num_epochs,
		batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		lora_r=args.lora_r,
		use_4bit=args.use_4bit,
		enable_full_training=not args.test_mode,
		local_files_only=args.local_only,
		generation_temperature=args.temperature,
		validate_augmented_data=not args.no_validation,
		max_length=args.max_length
	)
	
	set_seed(config.seed)
	
	config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
	force_print(TableLogger.create_simple_table("PIPELINE CONFIGURATION", config_dict))
	
	try:
		# Load data
		processor = AuditRecommendationDataProcessor(config)
		dataset_dict = processor.load_data()
		
		# Initialize model
		generator = LLMRecommendationGenerator(config)
		
		# Train
		trainer = generator.train(
			dataset_dict=dataset_dict,
			resume_from_checkpoint=args.resume_from
		)
		
		# Test generation
		force_print("\n" + "="*100)
		force_print("TESTING FINE-TUNED LLM")
		force_print("="*100)
		
		for i in range(min(3, len(processor.test_df))):
			test_constat = processor.test_df.iloc[i]['constat']
			test_reference = processor.test_df.iloc[i]['recommandations']
			
			generated = generator.generate_recommendations(test_constat, validate=True)
			
			# Compute structure score
			structure = generator.metrics_computer.compute_structure_score(generated)
			
			example_data = {
				"Observation": test_constat[:200] + "..." if len(test_constat) > 200 else test_constat,
				"Generated": generated[:300] + "..." if len(generated) > 300 else generated,
				"Reference": test_reference[:200] + "..." if len(test_reference) > 200 else test_reference,
				"Word Count": len(generated.split()),
				"Structure Score": f"{structure:.2f}"
			}
			
			force_print(TableLogger.create_simple_table(f"TEST EXAMPLE {i + 1}", example_data))
		
		final_summary = {
			"Training": "Complete",
			"Model location": config.output_dir,
			"Base model": config.model_path,
			"LoRA rank": config.lora_r,
			"Temperature": config.generation_temperature,
			"System": "Windows with 48GB VRAM",
			"Status": "Ready for inference"
		}
		force_print(TableLogger.create_simple_table("PIPELINE COMPLETE", final_summary))
		
	except Exception as e:
		logger.error(f"Pipeline failed: {e}")
		logger.error(f"Full traceback: {traceback.format_exc()}")
		raise


if __name__ == "__main__":
	main()
