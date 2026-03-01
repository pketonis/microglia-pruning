"""Inference backends for production serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .system import MicrogliaPruningSystem


class InferenceBackendError(RuntimeError):
    """Raised when an inference backend cannot be initialized."""


@dataclass
class GenerationConfig:
    """Common generation settings for both HF and vLLM backends."""

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


class HuggingFaceBackend:
    """Minimal HuggingFace text generation backend."""

    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        config = config or GenerationConfig()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


class VLLMBackend:
    """vLLM backend for high-throughput, continuous batching inference."""

    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise InferenceBackendError(
                "vLLM is not installed. Install with `pip install vllm`."
            ) from exc

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        config = config or GenerationConfig()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [out.outputs[0].text for out in outputs]


class InferenceEngine:
    """Backend selector that can call the pruning system directly.

    When ``pruning_system`` is provided, generation is delegated to
    :class:`src.system.MicrogliaPruningSystem` so dynamic budgeting and
    pruning masks are applied consistently.
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "vllm",
        tensor_parallel_size: int = 1,
        pruning_system: Optional[MicrogliaPruningSystem] = None,
    ):
        self.pruning_system = pruning_system
        backend = backend.lower()
        self.backend_name = backend

        if self.pruning_system is not None:
            self.backend = None
            return

        if backend == "vllm":
            self.backend = VLLMBackend(
                model_name=model_name,
                tensor_parallel_size=tensor_parallel_size,
            )
        elif backend == "hf":
            self.backend = HuggingFaceBackend(model_name=model_name)
        else:
            raise ValueError("backend must be one of {'vllm', 'hf'}")

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        use_pruning: bool = True,
        budget_keep_ratio: Optional[float] = None,
    ) -> str:
        if self.pruning_system is not None:
            config = config or GenerationConfig()
            return self.pruning_system.generate(
                prompt,
                max_new_tokens=config.max_new_tokens,
                use_pruning=use_pruning,
                budget_keep_ratio=budget_keep_ratio,
            )

        if isinstance(self.backend, VLLMBackend):
            return self.backend.generate_batch([prompt], config=config)[0]
        return self.backend.generate(prompt, config=config)
