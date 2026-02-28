"""Main pruning system that orchestrates training and inference."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import re
import os
import gc

from .agent import MicrogliaAgent
from .hooks import register_hooks, remove_hooks
from .pruned_attention import PrunedAttention
from .loss import compute_pruning_loss, get_alpha_schedule, compute_efficiency_metrics
from .utils import get_model_layers, setup_logging


class MicrogliaPruningSystem:
    """Orchestrator for the Microglia-inspired dynamic pruning system.

    This class manages the lifecycle of the pruning system, including model
    loading, agent initialization, training with curriculum learning, and
    rigorous evaluation.
    """
    
    def __init__(self,
                 model: str or nn.Module,
                 num_heads: int = 32,
                 hidden_dim: int = 128,
                 temperature: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 seed: int = 42):
        """Initializes the MicrogliaPruningSystem.

        Args:
            model (str or nn.Module): HuggingFace model name or a pre-loaded model.
            num_heads (int): Number of attention heads in the base model.
            hidden_dim (int): Hidden dimension for the MicrogliaAgents.
            temperature (float): Initial temperature for agent sigmoid masks.
            device (str): Device to place the model and agents on.
            seed (int): Random seed for reproducibility.
        """
        
        self.device = device
        self.num_heads = num_heads
        self.current_masks = {}
        self.pruning_enabled = False
        self.logger = setup_logging("MicrogliaSystem")
        
        self.logger.info(f"Initializing MicrogliaPruningSystem on {device}...")
        
        if isinstance(model, str):
            self.logger.info(f"Loading base model: {model}")
            
            config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            
            # Fix rope_scaling
            if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                if isinstance(config.rope_scaling, dict):
                    if 'type' not in config.rope_scaling:
                        config.rope_scaling = None
                        self.logger.warning("Disabled rope_scaling (missing type)")
                    elif config.rope_scaling.get('type') not in ['linear', 'dynamic', 'longrope']:
                        config.rope_scaling = None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                config=config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            
            # Fix Phi-3 EOS token issue
            if 'phi-3' in model.lower():
                self.logger.info("Fixing Phi-3 EOS token issue...")
                self.tokenizer.eos_token = "<|end|>"
                self.tokenizer.pad_token = "<|end|>" 
                eos_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
                self.model.config.eos_token_id = eos_id
                self.model.config.pad_token_id = eos_id
                # Also set in generation config if it exists
                if hasattr(self.model, 'generation_config'):
                    self.model.generation_config.eos_token_id = eos_id
                    self.model.generation_config.pad_token_id = eos_id
                self.logger.info(f"Set EOS token to '<|end|>' (ID: {eos_id})")
            else:
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.model = model
            self.tokenizer = None
        
        # Determine layers
        layers = self.get_layers()
        self.logger.info(f"Model has {len(layers)} layers")
        
        self.logger.info(f"Initializing {len(layers)} pruning agents...")
        self.agents = nn.ModuleList([
            MicrogliaAgent(hidden_dim, num_heads, temperature)
            for _ in range(len(layers))
        ])
        self.agents.to(device)
        
        # Don't wrap initially
        self.wrapped = False
        self.lora_applied = False
        
        self.activation_cache = {}
        self.training_history = []
        
        self.logger.info("System initialized successfully!")
        self.logger.info("Note: Pruning is DISABLED until training starts")
    
    def _apply_lora(self):
        """Apply LoRA BEFORE wrapping attention."""
        if self.lora_applied:
            return
        
        print("Applying LoRA for parameter-efficient training...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.lora_applied = True
    
    def get_layers(self):
        """Get the layers of the model, handling PEFT wrapping."""
        return get_model_layers(self.model)

    def _wrap_attention_layers(self):
        """Replace standard attention with pruned attention."""
        if self.wrapped:
            return
        
        layers = self.get_layers()
        print("Wrapping attention layers with pruning modules...")
        for idx, layer in enumerate(layers):
            original_attn = layer.self_attn
            layer.self_attn = PrunedAttention(
                original_attn, 
                self.agents[idx],
                hard_prune=False
            )
            layer.self_attn.enable_pruning = False
        self.wrapped = True
    
    def _enable_pruning(self, enable: bool = True):
        """Enable or disable pruning in all layers."""
        self.pruning_enabled = enable
        if not self.wrapped:
            return
        
        layers = self.get_layers()
        for layer in layers:
            if isinstance(layer.self_attn, PrunedAttention):
                layer.self_attn.enable_pruning = enable
        
        print(f"Pruning {'ENABLED' if enable else 'DISABLED'}")

    def set_hard_prune(self, enable: bool = True):
        """Enable or disable hard thresholding for pruning (inference)."""
        if not self.wrapped:
            print("Warning: Layers not yet wrapped. Hard prune will be set during wrapping.")
            return

        layers = self.get_layers()
        for layer in layers:
            if isinstance(layer.self_attn, PrunedAttention):
                layer.self_attn.hard_prune = enable

        print(f"Hard pruning {'ENABLED' if enable else 'DISABLED'}")
    
    def train(self,
             dataset_name: str = "gsm8k",
             num_epochs: int = 10,
             batch_size: int = 2,
             learning_rate: float = 1e-4,
             alpha_schedule: Tuple[float, float] = (0.01, 0.3),
             use_lora: bool = True,
             max_steps_per_epoch: int = 30,
             val_split: float = 0.1,
             early_stopping_patience: int = 3):
        """Train the pruning agents on a reasoning dataset with validation and checkpointing.

        Args:
            dataset_name: Name of the dataset to train on.
            num_epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate for pruning agents.
            alpha_schedule: Tuple of (alpha_min, alpha_max) for curriculum learning.
            use_lora: Whether to use LoRA for the base model.
            max_steps_per_epoch: Maximum number of steps per epoch.
            val_split: Fraction of data to use for validation.
            early_stopping_patience: Number of epochs to wait for validation improvement.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("Starting Training")
        self.logger.info("="*60)
        
        # Ensure base model doesn't store gradients to save memory
        # We only optimize the pruning agents
        self.model.requires_grad_(False)

        # Apply LoRA first if requested (before wrapping)
        if use_lora and not self.lora_applied:
            self._apply_lora()
        
        # Then wrap attention
        if not self.wrapped:
            self._wrap_attention_layers()
        
        # Ensure agents are trainable (in case they were frozen by self.model.requires_grad_)
        self.agents.requires_grad_(True)

        # Enable pruning for training
        self._enable_pruning(True)
        
        self.logger.info(f"Loading {dataset_name} dataset...")
        dataset = load_dataset(dataset_name, "main")
        
        self.logger.info("Preprocessing dataset...")
        # Use more data if available for rigorous training
        total_samples = min(1000, len(dataset['train']))
        full_subset = dataset['train'].select(range(total_samples))

        # Split into train and val
        indices = list(range(total_samples))
        import random
        random.shuffle(indices)
        split_idx = int(total_samples * (1 - val_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_subset = full_subset.select(train_indices)
        val_subset = full_subset.select(val_indices)
        
        # Process in batches to avoid OOM
        processed_examples = []
        batch_size_preprocess = 100
        
        for i in range(0, len(train_subset), batch_size_preprocess):
            batch = train_subset[i:i+batch_size_preprocess]
            prompts = [
                f"Question: {q}\nAnswer: {a}"
                for q, a in zip(batch['question'], batch['answer'])
            ]
            encoded = self.tokenizer(
                prompts,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            for j in range(len(prompts)):
                processed_examples.append({
                    'input_ids': encoded['input_ids'][j],
                    'attention_mask': encoded['attention_mask'][j]
                })
            
            del encoded
            gc.collect()
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, examples):
                self.examples = examples
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                return self.examples[idx]
        
        train_dataset = SimpleDataset(processed_examples)
        self.logger.info(f"Prepared {len(train_dataset)} training examples")
        
        self.logger.info("\nSetting up optimizer and scheduler for pruning agents...")
        optimizer = torch.optim.AdamW(
            self.agents.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        # Add a learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        self.model.train()
        alpha_min, alpha_max = alpha_schedule
        
        self.logger.info(f"\nTraining for {num_epochs} epochs...")
        self.logger.info(f"Alpha schedule: {alpha_min} -> {alpha_max}\n")
        
        best_val_loss = float('inf')
        best_agents_state = None
        patience_counter = 0

        for epoch in range(num_epochs):
            alpha = get_alpha_schedule(epoch, num_epochs, alpha_min, alpha_max)
            epoch_metrics = {'task_loss': 0.0, 'sparsity_loss': 0.0, 'total_loss': 0.0}
            
            print(f"\nEpoch {epoch+1}/{num_epochs} (alpha={alpha:.3f})")
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch, labels=batch['input_ids'])
                task_loss = outputs.loss
                
                all_masks = []
                layers = self.get_layers()
                for layer in layers:
                    if hasattr(layer.self_attn, 'last_masks') and layer.self_attn.last_masks is not None:
                        all_masks.append(layer.self_attn.last_masks)
                
                if all_masks:
                    masks = torch.cat(all_masks, dim=0)
                    loss_dict = compute_pruning_loss(task_loss, masks, alpha=alpha)
                    total_loss = loss_dict['total_loss']
                    
                    optimizer.zero_grad()
                    self.model.zero_grad()  # Also clear LoRA gradients
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agents.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_metrics['task_loss'] += loss_dict['task_loss']
                    epoch_metrics['sparsity_loss'] += loss_dict['sparsity_loss']
                    epoch_metrics['total_loss'] += total_loss.item()
                    
                    progress_bar.set_postfix({
                        'loss': f"{total_loss.item():.3f}",
                        'sparsity': f"{loss_dict['sparsity_loss']:.3f}"
                    })
                
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                
                if step >= max_steps_per_epoch:
                    break
            
            # Update learning rate
            scheduler.step()

            n_steps = min(len(train_loader), max_steps_per_epoch)
            avg_metrics = {k: v/n_steps for k, v in epoch_metrics.items()}
            self.logger.info(f"Epoch {epoch+1} Summary:")
            self.logger.info(f"  Task Loss: {avg_metrics['task_loss']:.4f}")
            self.logger.info(f"  Sparsity Loss: {avg_metrics['sparsity_loss']:.4f}")
            self.logger.info(f"  Total Loss: {avg_metrics['total_loss']:.4f}")
            self.logger.info(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
            
            self.training_history.append(avg_metrics)

            # Simple validation (using a small subset of val data)
            self.model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for i in range(min(5, len(val_subset))):
                    item = val_subset[i]
                    prompt = f"Question: {item['question']}\nAnswer: {item['answer']}"
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
                    outputs = self.model(**inputs, labels=inputs['input_ids'])
                    val_loss += outputs.loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')
            self.logger.info(f"  Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                import copy
                best_agents_state = copy.deepcopy(self.agents.state_dict())
                patience_counter = 0
                self.logger.info("  New best model found!")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            self.model.train()
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # Restore best weights if found
        if best_agents_state is not None:
            self.logger.info("Restoring best agent weights from training...")
            self.agents.load_state_dict(best_agents_state)

        # Disable pruning after training
        self._enable_pruning(False)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Complete!")
        self.logger.info("="*60)
    
    def generate(self, prompt: str, max_new_tokens: int = 256, use_pruning: bool = None, **kwargs):
        """Generate text with optional pruning."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer available")
        
        # Use current system state if not specified
        if use_pruning is None:
            use_pruning = self.pruning_enabled

        # Ensure pruning state is correct
        self._enable_pruning(use_pruning)
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,  # Disable temperature for greedy
                top_p=None,
                use_cache=False,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def get_sparsity(self) -> float:
        """Calculate current average pruning sparsity."""
        if not self.wrapped:
            return 0.0
        
        all_masks = []
        layers = self.get_layers()
        for layer in layers:
            if hasattr(layer.self_attn, 'last_masks') and layer.self_attn.last_masks is not None:
                all_masks.append(layer.self_attn.last_masks)
        
        if not all_masks:
            return 0.0
        
        masks = torch.cat(all_masks, dim=0)
        metrics = compute_efficiency_metrics(masks)
        return metrics['sparsity']
    
    def evaluate(self,
                 dataset_name: str = "gsm8k",
                 split: str = "test",
                 max_samples: int = 200,
                 use_pruning: bool = True,
                 num_bootstrap: int = 1000) -> Dict:
        """Evaluate accuracy on a reasoning benchmark with bootstrap confidence intervals.

        Args:
            dataset_name: Name of the dataset to evaluate on.
            split: Dataset split to use.
            max_samples: Maximum number of samples to evaluate.
            use_pruning: Whether to enable pruning during evaluation.
            num_bootstrap: Number of bootstrap resamples for CI calculation.

        Returns:
            results: Dictionary containing accuracy, CIs, and other metrics.
        """
        self.logger.info(f"Evaluating on {dataset_name} ({split} split)...")
        
        # Use provided pruning setting (defaults to True for pruned eval)
        self._enable_pruning(use_pruning)
        
        dataset = load_dataset(dataset_name, "main", split=split)
        
        self.model.eval()
        results_list = []
        
        if max_samples is None:
            max_samples = len(dataset)

        progress_bar = tqdm(dataset.select(range(min(max_samples, len(dataset)))), desc="Evaluating")
        
        with torch.no_grad():
            for example in progress_bar:
                prompt = f"Question: {example['question']}\nAnswer:"
                output = self.generate(prompt, max_new_tokens=256)
                
                gold_answer = self._extract_answer(example['answer'])
                pred_answer = self._extract_answer(output)
                
                is_correct = 0
                if pred_answer is not None and gold_answer is not None:
                    if abs(pred_answer - gold_answer) < 0.01:
                        is_correct = 1
                
                results_list.append(is_correct)

                current_acc = sum(results_list) / len(results_list)
                progress_bar.set_postfix({'accuracy': f"{current_acc:.1%}"})

        accuracy = sum(results_list) / len(results_list) if results_list else 0.0

        # Bootstrap Confidence Intervals
        self.logger.info(f"Calculating bootstrap confidence intervals ({num_bootstrap} resamples)...")
        boot_accuracies = []
        import numpy as np
        results_arr = np.array(results_list)
        for _ in range(num_bootstrap):
            if len(results_arr) > 0:
                resample = np.random.choice(results_arr, size=len(results_arr), replace=True)
                boot_accuracies.append(resample.mean())
            else:
                boot_accuracies.append(0.0)

        ci_lower = np.percentile(boot_accuracies, 2.5)
        ci_upper = np.percentile(boot_accuracies, 97.5)
        
        sparsity = self.get_sparsity()
        
        results = {
            'accuracy': accuracy,
            'ci_95': (ci_lower, ci_upper),
            'correct': sum(results_list),
            'total': len(results_list),
            'sparsity': sparsity
        }
        
        self.logger.info(f"Results:")
        self.logger.info(f"  Accuracy: {accuracy:.2%} [95% CI: {ci_lower:.2%} - {ci_upper:.2%}]")
        self.logger.info(f"  Correct: {results['correct']}/{results['total']}")
        self.logger.info(f"  Sparsity: {sparsity:.1%}")
        
        return results
    
    def _extract_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        if '####' in text:
            text = text.split('####')[1]
        
        numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
        
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return None
        return None
    
    def save(self, path: str):
        """Save the trained pruning system."""
        print(f"Saving model to {path}...")
        torch.save({
            'agents': self.agents.state_dict(),
            'training_history': self.training_history,
        }, path)
        print("Saved successfully!")
    
    def load(self, path: str):
        """Load a trained pruning system."""
        print(f"Loading model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.agents.load_state_dict(checkpoint['agents'])
        self.training_history = checkpoint.get('training_history', [])
        print("Loaded successfully!")
