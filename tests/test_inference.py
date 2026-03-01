from unittest.mock import patch

import pytest

from src.inference import GenerationConfig, InferenceBackendError, InferenceEngine


class DummyVLLMBackend:
    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size

    def generate_batch(self, prompts, config=None):
        return [f"vllm:{prompt}" for prompt in prompts]


class DummyHFBackend:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt, config=None):
        return f"hf:{prompt}"


def test_inference_engine_vllm_path():
    with patch("src.inference.VLLMBackend", DummyVLLMBackend):
        engine = InferenceEngine(model_name="dummy", backend="vllm")
        assert engine.generate("hello") == "vllm:hello"


def test_inference_engine_hf_path():
    with patch("src.inference.HuggingFaceBackend", DummyHFBackend):
        engine = InferenceEngine(model_name="dummy", backend="hf")
        assert engine.generate("hello") == "hf:hello"


def test_generation_config_defaults():
    cfg = GenerationConfig()
    assert cfg.max_new_tokens == 128
    assert cfg.top_p == 1.0


def test_backend_validation_error():
    with pytest.raises(ValueError):
        InferenceEngine(model_name="dummy", backend="invalid")


def test_vllm_import_error_message():
    import builtins
    from src.inference import VLLMBackend

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "vllm":
            raise ImportError("no module")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(InferenceBackendError):
            VLLMBackend("dummy")


class DummyPruningSystem:
    def generate(self, prompt, max_new_tokens=128, use_pruning=True, budget_keep_ratio=None):
        return f"pruned:{prompt}:{max_new_tokens}:{use_pruning}:{budget_keep_ratio}"


def test_inference_engine_pruning_system_path():
    engine = InferenceEngine(model_name="dummy", pruning_system=DummyPruningSystem())
    out = engine.generate("hello", config=GenerationConfig(max_new_tokens=32), budget_keep_ratio=0.6)
    assert out.startswith("pruned:hello:32:True:0.6")
