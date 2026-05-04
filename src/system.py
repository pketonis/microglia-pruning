"""
Microglia-Inspired Dynamic Pruning for Reasoning Models

Authors: Tommaso R. Marena (The Catholic University of America)
         Panos Ketonis (Yale University)

Copyright (c) 2026
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Tuple, Dict
from contextlib import nullcontext
from tqdm import tqdm
import re
import os
import gc

from .agent import MicrogliaAgent
from .hooks import register_hooks, remove_hooks
from .pruned_attention import PrunedAttention
from .loss import compute_pruning_loss, get_alpha_schedule, compute_efficiency_metrics
from .utils import get_model_layers, setup_logging
from .budget import DynamicPruningBudget
from .model_registry import resolve_model_spec


def _sanitize_rope_scaling(config, logger=None) -> None:
    """Null out a malformed rope_scaling dict in-place.

    Phi-3 (and several other HuggingFace models) ship with a rope_scaling
    config that is missing the required 'type' key. Their custom modeling
    code reads config.rope_scaling['type'] directly inside __init__, which
    raises a KeyError before from_pretrained even returns. Setting the
    attribute to None is the safest fix — the model falls back to standard
    RoPE.

    We skip this sanitization for models that use the newer rope_parameters
    field (e.g. Qwen3), where modifying rope_scaling would corrupt the setup.
    """
    uses_rope_parameters = (
        hasattr(config, 'rope_parameters') and config.rope_parameters is not None
    )
    if uses_rope_parameters:
        return

    if not (hasattr(config, 'rope_scaling') and config.rope_scaling is not None):
        return

    if not isinstance(config.rope_scaling, dict):
        return

    rope_type = config.rope_scaling.get('type') or config.rope_scaling.get('rope_type')
    valid_types = {'linear', 'dynamic', 'longrope', 'yarn', 'su', 'llama3'}

    if rope_type is None or rope_type not in valid_types:
        if logger:
            logger.warning(
                f"rope_scaling has unsupported/missing type '{rope_type}' — "
                "setting rope_scaling=None so model falls back to standard RoPE."
            )
        config.rope_scaling = None


def _mask_labels_for_padding(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Return a labels tensor with pad positions replaced by -100.

    HuggingFace CausalLM models pass labels directly to cross-entropy.
    Padding positions must be set to -100 (the ignore_index) so they are
    excluded from the loss — otherwise NaN propagates whenever a batch row
    is dominated by pad tokens (common with short sequences and large
    max_length).
    """
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return labels


class MicrogliaPruningSystem:
    """Orchestrator for the Microglia-inspired dynamic pruning system."""

    def __init__(self,
                 model: str or nn.Module,
                 num_heads: int = 32,
                 hidden_dim: int = 128,
                 temperature: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 seed: int = 42):
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.device = device
        self.num_heads = num_heads
        self.temperature = float(temperature)
        self.current_masks = {}
        self.pruning_enabled = False
        self.logger = setup_logging("MicrogliaSystem")
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.logger.info(f"Initializing MicrogliaPruningSystem on {device}...")

        resolved_spec = resolve_model_spec(model if isinstance(model, str) else "custom")

        if isinstance(model, str):
            model_name = resolved_spec.name
            self.logger.info(f"Loading base model: {model}")

            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

            # Patch malformed rope_scaling before it reaches model __init__.
            # Must be done on the config object itself — mutating it after
            # from_pretrained is too late for models with custom modeling code.
            _sanitize_rope_scaling(config, self.logger)

            load_kwargs = dict(
                config=config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="eager",
                trust_remote_code=True,
            )

            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            except (KeyError, ValueError) as exc:
                # Belt-and-suspenders: if the model's custom __init__ still
                # trips on rope_scaling (e.g. because it re-reads the raw
                # HuggingFace config cache), retry with rope_scaling disabled.
                self.logger.warning(
                    f"from_pretrained raised {type(exc).__name__}: {exc}. "
                    "Retrying with rope_scaling explicitly disabled on config."
                )
                config.rope_scaling = None
                # Some models also expose this via model_kwargs
                load_kwargs["rope_scaling"] = None
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            if 'phi-3' in model_name.lower() or 'phi3' in model_name.lower():
                self.logger.info("Applying Phi-3 EOS/PAD token fix...")
                self.tokenizer.eos_token = "<|end|>"
                self.tokenizer.pad_token = "<|end|>"
                eos_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
                self.model.config.eos_token_id = eos_id
                self.model.config.pad_token_id = eos_id
                if hasattr(self.model, 'generation_config'):
                    self.model.generation_config.eos_token_id = eos_id
                    self.model.generation_config.pad_token_id = eos_id
                self.logger.info(f"Set EOS/PAD token to '<|end|>' (ID: {eos_id})")
            else:
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id

            # Detect whether this tokenizer has a chat template.
            # Instruct models (Phi-3-Instruct, Qwen-Instruct, Llama-Instruct)
            # require apply_chat_template to produce coherent outputs.
            self._has_chat_template = (
                hasattr(self.tokenizer, 'apply_chat_template')
                and self.tokenizer.chat_template is not None
            )
            if self._has_chat_template:
                self.logger.info("Chat template detected — prompts will use apply_chat_template.")
            else:
                self.logger.info("No chat template — using plain-text prompt format.")
        else:
            self.model = model
            self.tokenizer = None
            self._has_chat_template = False

        layers = self.get_layers()
        detected_num_heads = getattr(getattr(self.model, "config", object()), "num_attention_heads", num_heads)
        self.num_heads = int(detected_num_heads or num_heads)
        self.logger.info(f"Model has {len(layers)} layers")
        self.logger.info(f"Detected attention heads: {self.num_heads}")

        self.logger.info(f"Initializing {len(layers)} pruning agents...")
        total_layers = len(layers)
        self.agents = nn.ModuleList([
            MicrogliaAgent(hidden_dim, self.num_heads, temperature, num_layers=total_layers, layer_idx=layer_idx)
            for layer_idx in range(total_layers)
        ])
        self.agents.to(device)

        self.wrapped = False
        self.lora_applied = False
        self.activation_cache = {}
        self.training_history = []
        self.budget_controller = DynamicPruningBudget()
        self.last_budget = None

        self.logger.info("System initialized successfully!")
        self.logger.info("Note: Pruning is DISABLED until training starts")

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _format_prompt(self, question: str, include_answer_prefix: bool = True) -> str:
        """Format a GSM8K question into the correct prompt string for this model."""
        if self._has_chat_template:
            messages = [{"role": "user", "content": f"Solve this math problem step by step.\n\n{question}"}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=include_answer_prefix
            )
        else:
            return f"Question: {question}\nAnswer:"

    def _format_train_sample(self, question: str, answer: str) -> str:
        """Format a full Q+A pair for training (teacher-forcing)."""
        if self._has_chat_template:
            messages = [
                {"role": "user", "content": f"Solve this math problem step by step.\n\n{question}"},
                {"role": "assistant", "content": answer}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            return f"Question: {question}\nAnswer: {answer}"

    # ------------------------------------------------------------------
    # LoRA / attention wrapping
    # ------------------------------------------------------------------

    def _apply_lora(self):
        """Apply LoRA BEFORE wrapping attention."""
        if self.lora_applied:
            return
        print("Applying LoRA for parameter-efficient training...")
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        model_name = getattr(self.model.config, "_name_or_path", "").lower()
        if "gpt2" in model_name:
            target_modules = ["c_attn", "c_proj"]
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.lora_applied = True

    def get_layers(self):
        return get_model_layers(self.model)

    def _wrap_attention_layers(self):
        if self.wrapped:
            return
        layers = self.get_layers()
        print("Wrapping attention layers with pruning modules...")
        for idx, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
            if attn is not None:
                new_attn = PrunedAttention(attn, self.agents[idx], hard_prune=False)
                new_attn.enable_pruning = False
                if hasattr(layer, "self_attn"):
                    layer.self_attn = new_attn
                else:
                    layer.attn = new_attn
            else:
                raise AttributeError(f"Could not find attention module in layer {type(layer)}")
        self.wrapped = True

    def _enable_pruning(self, enable: bool = True):
        self.pruning_enabled = enable
        if not self.wrapped:
            return
        for layer in self.get_layers():
            attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
            if isinstance(attn, PrunedAttention):
                attn.enable_pruning = enable
        print(f"Pruning {'ENABLED' if enable else 'DISABLED'}")

    def _set_budget_keep_ratio(self, keep_ratio: Optional[float]) -> None:
        self.last_budget = keep_ratio
        if not self.wrapped:
            return
        for layer in self.get_layers():
            attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
            if isinstance(attn, PrunedAttention):
                attn.set_budget_keep_ratio(keep_ratio)

    def set_hard_prune(self, enable: bool = True):
        if not self.wrapped:
            print("Warning: Layers not yet wrapped.")
            return
        for layer in self.get_layers():
            attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
            if isinstance(attn, PrunedAttention):
                attn.hard_prune = enable
        print(f"Hard pruning {'ENABLED' if enable else 'DISABLED'}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self,
              dataset_name: str = "gsm8k",
              num_epochs: int = 10,
              batch_size: int = 2,
              learning_rate: float = 1e-4,
              alpha_schedule: Tuple[float, float] = (0.01, 0.3),
              alpha_schedule_type: str = "linear",
              use_lora: bool = True,
              max_steps_per_epoch: int = 30,
              val_split: float = 0.1,
              early_stopping_patience: int = 3,
              precision: str = "fp32",
              use_budget: bool = True,
              max_val_samples: int = 50,
              max_length: int = 256):
        self.logger.info("\n" + "="*60)
        self.logger.info("Starting Training")
        self.logger.info("="*60)

        self.model.requires_grad_(False)
        if use_lora and not self.lora_applied:
            self._apply_lora()
        if not self.wrapped:
            self._wrap_attention_layers()
        self.agents.requires_grad_(True)
        self._enable_pruning(True)

        if use_budget:
            self.logger.info("Dynamic pruning budget is ENABLED for training")
        else:
            self.logger.info("Dynamic pruning budget is DISABLED for training")

        self.logger.info(f"Loading {dataset_name} dataset...")
        dataset = load_dataset(dataset_name, "main")

        self.logger.info("Preprocessing dataset...")
        total_samples = min(1000, len(dataset['train']))
        full_subset = dataset['train'].select(range(total_samples))

        indices = list(range(total_samples))
        import random
        random.Random(self.seed if hasattr(self, 'seed') else 42).shuffle(indices)
        split_idx = int(total_samples * (1 - val_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_subset = full_subset.select(train_indices)
        val_subset = full_subset.select(val_indices)

        processed_examples = []
        truncated_examples = 0
        total_tokenized_examples = 0
        batch_size_preprocess = 100

        pad_token_id = self.tokenizer.pad_token_id

        for i in range(0, len(train_subset), batch_size_preprocess):
            batch = train_subset[i:i+batch_size_preprocess]
            prompts = [
                self._format_train_sample(q, a)
                for q, a in zip(batch['question'], batch['answer'])
            ]
            encoded = self.tokenizer(
                prompts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            attention_mask = encoded['attention_mask']
            total_tokenized_examples += attention_mask.size(0)
            truncated_examples += int((attention_mask.sum(dim=1) >= max_length).sum().item())
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
        if total_tokenized_examples > 0:
            truncation_fraction = truncated_examples / total_tokenized_examples
            if truncation_fraction > 0.1:
                self.logger.warning(
                    f"WARNING: {truncation_fraction * 100:.1f}% of training examples were truncated "
                    f"at max_length={max_length}. Consider increasing max_length."
                )

        self.logger.info("\nSetting up optimizer and scheduler...")
        optimizer = torch.optim.AdamW(self.agents.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        self.model.train()
        alpha_min, alpha_max = alpha_schedule

        if precision not in {'fp32', 'fp16', 'bf16'}:
            raise ValueError("precision must be one of {'fp32', 'fp16', 'bf16'}")
        amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[precision]
        scaler = torch.amp.GradScaler('cuda', enabled=(precision == 'fp16' and torch.cuda.is_available()))

        self.logger.info(f"\nTraining for {num_epochs} epochs...")
        self.logger.info(f"Alpha schedule ({alpha_schedule_type}): {alpha_min} -> {alpha_max}\n")

        best_val_loss = float('inf')
        best_agents_state = None
        patience_counter = 0

        for epoch in range(num_epochs):
            alpha = get_alpha_schedule(epoch, num_epochs, alpha_min, alpha_max, schedule_type=alpha_schedule_type)
            epoch_metrics = {'task_loss': 0.0, 'sparsity_loss': 0.0, 'total_loss': 0.0}
            print(f"\nEpoch {epoch+1}/{num_epochs} (alpha={alpha:.3f})")

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            actual_steps = 0

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Mask pad positions in labels so cross-entropy ignores them.
                # Without this, loss over all-pad rows produces NaN.
                labels = _mask_labels_for_padding(batch['input_ids'], pad_token_id)

                if use_budget:
                    prompt_preview = self.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
                    keep_ratio = self.budget_controller.compute_keep_ratio(prompt_preview)
                    self._set_budget_keep_ratio(keep_ratio)
                else:
                    self._set_budget_keep_ratio(None)

                amp_context = (
                    torch.autocast(device_type='cuda', dtype=amp_dtype)
                    if precision in {'fp16', 'bf16'} and torch.cuda.is_available()
                    else (
                        torch.autocast(device_type='cpu', dtype=torch.bfloat16)
                        if precision == 'bf16'
                        else nullcontext()
                    )
                )
                with amp_context:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=labels,
                    )
                    task_loss = outputs.loss

                all_masks = []
                for layer in self.get_layers():
                    attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
                    if hasattr(attn, 'last_masks') and attn.last_masks is not None:
                        all_masks.append(attn.last_masks)

                if all_masks:
                    masks = torch.cat(all_masks, dim=0)
                    loss_dict = compute_pruning_loss(task_loss, masks, alpha=alpha)
                    total_loss = loss_dict['total_loss']

                    optimizer.zero_grad()
                    self.model.zero_grad()
                    if scaler.is_enabled():
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.agents.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
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
                    actual_steps += 1

                if step % 10 == 0:
                    torch.cuda.empty_cache()
                if step + 1 >= max_steps_per_epoch:
                    break

            scheduler.step()
            avg_metrics = {k: v/max(actual_steps, 1) for k, v in epoch_metrics.items()}
            self.logger.info(f"Epoch {epoch+1} Summary:")
            self.logger.info(f"  Task Loss: {avg_metrics['task_loss']:.4f}")
            self.logger.info(f"  Sparsity Loss: {avg_metrics['sparsity_loss']:.4f}")
            self.logger.info(f"  Total Loss: {avg_metrics['total_loss']:.4f}")
            self.logger.info(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
            self.training_history.append(avg_metrics)

            # Validation — also mask labels to avoid NaN val loss
            self.model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for i in range(min(max_val_samples, len(val_subset))):
                    item = val_subset[i]
                    prompt = self._format_train_sample(item['question'], item['answer'])
                    inputs = self.tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=max_length
                    ).to(self.device)
                    val_labels = _mask_labels_for_padding(inputs['input_ids'], pad_token_id)
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=val_labels,
                    )
                    if not torch.isnan(outputs.loss):
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

        if best_agents_state is not None:
            self.logger.info("Restoring best agent weights from training...")
            self.agents.load_state_dict(best_agents_state)

        self._enable_pruning(False)
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Complete!")
        self.logger.info("="*60)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 use_pruning: bool = None,
                 budget_keep_ratio: Optional[float] = None,
                 apply_chat_template: bool = True,
                 **kwargs):
        """Generate text with optional pruning."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer available")

        if use_pruning is None:
            use_pruning = self.pruning_enabled
        self._enable_pruning(use_pruning)
        if use_pruning:
            if budget_keep_ratio is None:
                budget_keep_ratio = self.budget_controller.compute_keep_ratio(prompt)
            self._set_budget_keep_ratio(budget_keep_ratio)
        else:
            self._set_budget_keep_ratio(None)

        self.model.eval()

        if apply_chat_template:
            formatted = self._format_prompt(prompt, include_answer_prefix=True)
        else:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        input_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=False,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                **kwargs
            )

        # Decode only the newly generated tokens (strip echoed prompt)
        new_tokens = output_ids[0][input_len:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return generated_text

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self,
                 dataset_name: str = "gsm8k",
                 split: str = "test",
                 max_samples: int = 200,
                 use_pruning: bool = True,
                 num_bootstrap: int = 1000) -> Dict:
        """Evaluate accuracy with bootstrap confidence intervals."""
        self.logger.info(f"Evaluating on {dataset_name} ({split} split)...")
        self._enable_pruning(use_pruning)

        dataset = load_dataset(dataset_name, "main", split=split)
        self.model.eval()
        results_list = []

        if max_samples is None:
            max_samples = len(dataset)

        progress_bar = tqdm(
            dataset.select(range(min(max_samples, len(dataset)))),
            desc="Evaluating"
        )

        with torch.no_grad():
            for example in progress_bar:
                output = self.generate(example['question'], max_new_tokens=256)

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

        self.logger.info(f"Calculating bootstrap CIs ({num_bootstrap} resamples)...")
        import numpy as np
        results_arr = np.array(results_list)
        boot_accuracies = [
            np.random.choice(results_arr, size=len(results_arr), replace=True).mean()
            for _ in range(num_bootstrap)
        ]
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

        self.logger.info(f"  Accuracy: {accuracy:.2%} [95% CI: {ci_lower:.2%} - {ci_upper:.2%}]")
        self.logger.info(f"  Correct: {results['correct']}/{results['total']}")
        self.logger.info(f"  Sparsity: {sparsity:.1%}")
        return results

    # ------------------------------------------------------------------
    # Sparsity
    # ------------------------------------------------------------------

    def get_sparsity(self) -> float:
        if not self.wrapped:
            return 0.0
        all_masks = []
        for layer in self.get_layers():
            attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
            if hasattr(attn, 'last_masks') and attn.last_masks is not None:
                all_masks.append(attn.last_masks)
        if not all_masks:
            return 0.0
        masks = torch.cat(all_masks, dim=0)
        return compute_efficiency_metrics(masks)['sparsity']

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def _extract_answer(self, text: str) -> Optional[float]:
        """Extract a numerical answer from model output or gold text.

        Handles:
          - GSM8K gold delimiter (####)
          - Qwen3 / DeepSeek thinking-model output (<think>...</think> blocks)
          - Bold markdown final answer (**42**)
          - LaTeX boxed answer (\\boxed{42})
          - Natural language triggers ("the answer is", "= ", etc.)
          - Last-number fallback
        """
        # Strip Qwen3 / DeepSeek-style thinking traces before extraction.
        # The model wraps its reasoning in <think>...</think>; the actual
        # answer appears after </think>. If we search the full string we
        # pick up numbers from the reasoning steps instead.
        think_end = text.rfind('</think>')
        if think_end != -1:
            text = text[think_end + len('</think>'):]
        # Also strip any residual <think> opening that was never closed
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)

        cleaned = text.replace(',', '')
        unit_pattern = r"\b(?:dollars?|cents?|miles?|kg|students?|apples?|days?|hours?|minutes?|years?|meters?|feet|pounds?)\b"

        def _clean_segment(segment: str) -> str:
            segment = re.sub(r"[$\xa3\u20ac%]", "", segment)
            segment = re.sub(unit_pattern, "", segment, flags=re.IGNORECASE)
            return segment

        def _first_number(segment: str) -> Optional[float]:
            segment = _clean_segment(segment)
            match = re.search(r"-?\d+(?:\.\d+)?", segment)
            if not match:
                return None
            try:
                return float(match.group(0))
            except ValueError:
                return None

        # 1. GSM8K gold-answer delimiter
        if '####' in cleaned:
            value = _first_number(cleaned.split('####', 1)[1])
            if value is not None:
                return value

        # 2. LaTeX boxed answer: \boxed{42}  (DeepSeek, some Qwen outputs)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', cleaned)
        if boxed_match:
            value = _first_number(boxed_match.group(1))
            if value is not None:
                return value

        # 3. Bold markdown final answer: **42**  (Qwen3 instruct style)
        bold_match = re.search(r'\*\*(-?\d+(?:\.\d+)?)\*\*', cleaned)
        if bold_match:
            try:
                return float(bold_match.group(1))
            except ValueError:
                pass

        # 4. Natural-language triggers
        triggers = ["the answer is", "= ", "equals ", "therefore"]
        lowered = cleaned.lower()
        for trigger in triggers:
            idx = lowered.find(trigger)
            if idx != -1:
                value = _first_number(cleaned[idx + len(trigger):])
                if value is not None:
                    return value

        # 5. Last "Answer:" marker (handles echoed prompts)
        answer_idx = lowered.rfind("answer:")
        if answer_idx != -1:
            value = _first_number(cleaned[answer_idx + len("answer:"):])
            if value is not None:
                return value

        # 6. Last number fallback
        numbers = re.findall(r"-?\d+(?:\.\d+)?", _clean_segment(cleaned))
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return None
        return None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        print(f"Saving checkpoint to {path}...")
        payload = {
            'version': 2,
            'agents': self.agents.state_dict(),
            'training_history': self.training_history,
            'config': {
                'num_heads': self.num_heads,
                'temperature': self.temperature,
                'last_budget': self.last_budget,
                'pruning_enabled': self.pruning_enabled,
            },
        }
        if self.lora_applied:
            payload['lora'] = self.model.state_dict()
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        torch.save(payload, path)
        print("Saved successfully!")

    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, load_lora: bool = True):
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        if 'agents' not in checkpoint:
            raise KeyError("Checkpoint missing 'agents'.")

        fc1_weight_key = next((k for k in checkpoint['agents'] if k.endswith('fc1.weight')), None)
        if fc1_weight_key:
            ckpt_hidden_dim = checkpoint['agents'][fc1_weight_key].shape[0]
            current_hidden_dim = self.agents[0].fc1.out_features
            if ckpt_hidden_dim != current_hidden_dim:
                print(f"Hidden dim mismatch ({ckpt_hidden_dim} vs {current_hidden_dim}). Re-initializing agents...")
                total_layers = len(self.get_layers())
                self.agents = nn.ModuleList([
                    MicrogliaAgent(ckpt_hidden_dim, self.num_heads, self.temperature,
                                   num_layers=total_layers, layer_idx=i)
                    for i in range(total_layers)
                ])
                self.agents.to(self.device)
                if self.wrapped:
                    self.wrapped = False
                    self._wrap_attention_layers()

        self.agents.load_state_dict(checkpoint['agents'], strict=False)
        self.training_history = checkpoint.get('training_history', [])
        config = checkpoint.get('config', {})
        self.last_budget = config.get('last_budget', None)
        self.temperature = float(config.get('temperature', self.temperature))
        for agent in self.agents:
            agent.set_temperature(self.temperature)

        if load_lora:
            if 'lora' not in checkpoint and self.lora_applied:
                raise KeyError("Requested LoRA restore but checkpoint has no 'lora'.")
            if 'lora' in checkpoint:
                self.model.load_state_dict(checkpoint['lora'], strict=False)
        elif 'lora' in checkpoint and not self.lora_applied:
            print("Checkpoint contains LoRA weights. Applying LoRA adapters...")
            self._apply_lora()
            self.model.load_state_dict(checkpoint['lora'], strict=False)

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded successfully!")

    def save(self, path: str):
        self.save_checkpoint(path)

    def load(self, path: str):
        self.load_checkpoint(path, load_lora=False)
