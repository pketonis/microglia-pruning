from src.model_registry import resolve_model_spec
from src.rigor import bootstrap_ci


def test_model_registry_aliases():
    assert resolve_model_spec("phi3").name.endswith("phi-3-mini-4k-instruct")
    assert "Llama" in resolve_model_spec("llama3").name
    assert "Mistral" in resolve_model_spec("mistral").name


def test_rigor_package_importable():
    stats = bootstrap_ci([0, 1, 1, 0], num_bootstrap=50, ci=0.95)
    assert stats["ci_low"] <= stats["ci_high"]
