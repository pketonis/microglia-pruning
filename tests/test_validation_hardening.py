import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.agent import MicrogliaAgent
from src.budget import DynamicPruningBudget
from src.inference import GenerationConfig, InferenceEngine
from src.loss import compute_pruning_loss, get_alpha_schedule
from src.model_registry import resolve_model_spec
from src.precision import MixedPrecisionTrainer, PrecisionConfig
from src.pruned_attention import PrunedAttention
from src.rigor.statistics import bootstrap_ci, paired_bootstrap_test
from src.statistics import NUM_STATS_PER_HEAD, compute_layer_stats
from src.system import MicrogliaPruningSystem


class DummyAttention(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_heads = num_heads

    def forward(self, hidden_states, output_attentions=True, **kwargs):
        out = self.proj(hidden_states)
        b, s, _ = hidden_states.shape
        attn = torch.softmax(torch.randn(b, self.num_heads, s, s), dim=-1)
        return out, attn


def test_gradient_flow_batch_gt_one_and_multi_loss_terms():
    attn = DummyAttention()
    agent = MicrogliaAgent(hidden_dim=32, num_heads=4, temperature=1.0)
    module = PrunedAttention(attn, agent)
    module.enable_pruning = True

    x = torch.randn(3, 8, 64, requires_grad=True)
    out, _ = module(x)
    task_loss = out.pow(2).mean()
    loss = compute_pruning_loss(task_loss, module.last_masks, alpha=0.3, beta=0.1)["total_loss"]
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in agent.parameters())


def test_near_zero_masks_stability_keeps_one_head():
    class ZeroAgent(nn.Module):
        def forward(self, s):
            return torch.zeros(s.shape[0], 4)

    module = PrunedAttention(DummyAttention(), ZeroAgent())
    module.enable_pruning = True
    out, _ = module(torch.randn(2, 5, 64))
    assert out.shape == (2, 5, 64)
    assert torch.all(module.last_masks.sum(dim=1) >= 1.0)


def test_budget_adjust_constraints_and_mixed_batch():
    budget = DynamicPruningBudget()
    vals = budget.adjust(["hello", "Explain the full proof with 12 steps and symbols (a+b)/c"])
    assert len(vals) == 2
    assert all(0.35 <= v <= 0.95 for v in vals)
    assert vals[1] >= vals[0]


def test_budget_adjust_static_override_and_validation():
    budget = DynamicPruningBudget()
    vals = budget.adjust(["a", "b"], static_override=0.2)
    assert vals == [0.35, 0.35]
    with pytest.raises(ValueError):
        budget.adjust(["a"], static_override=1.5)


def test_statistics_entropy_uniform_and_peaked_and_zero_rows():
    b, h, s = 1, 4, 8
    hidden = torch.randn(b, s, h * 4)
    uniform = torch.full((b, h, s, s), 1 / s)
    peaked = torch.zeros((b, h, s, s))
    peaked[..., 0] = 1.0
    zeroish = torch.zeros((b, h, s, s))

    su = compute_layer_stats(hidden, uniform)
    sp = compute_layer_stats(hidden, peaked)
    sz = compute_layer_stats(hidden, zeroish)

    assert su.shape[-1] == NUM_STATS_PER_HEAD * h
    assert torch.isfinite(su).all() and torch.isfinite(sp).all() and torch.isfinite(sz).all()
    entropy_uniform = su[:, 2 * h : 3 * h].mean()
    entropy_peaked = sp[:, 2 * h : 3 * h].mean()
    assert entropy_uniform > entropy_peaked


@pytest.mark.parametrize("num_heads", [32, 40, 48])
def test_statistics_variable_head_counts(num_heads):
    hidden = torch.randn(2, 4, num_heads * 2)
    attn = torch.softmax(torch.randn(2, num_heads, 4, 4), dim=-1)
    stats = compute_layer_stats(hidden, attn)
    assert stats.shape == (2, NUM_STATS_PER_HEAD * num_heads)


def test_hard_prune_binary_masks_and_half_mask_loss_entropy():
    agent = MicrogliaAgent(hidden_dim=32, num_heads=4)
    module = PrunedAttention(DummyAttention(), agent, hard_prune=True)
    module.enable_pruning = True
    module.eval()
    _ = module(torch.randn(2, 5, 64))
    assert torch.all((module.last_masks == 0) | (module.last_masks == 1))

    masks = torch.full((3, 4), 0.5)
    out = compute_pruning_loss(torch.tensor(1.0), masks, alpha=0.1, beta=0.1)
    assert out["entropy_loss"] > 0


def test_alpha_schedules_and_single_epoch_jump():
    linear_mid = get_alpha_schedule(5, 10, 0.01, 0.3, schedule_type="linear")
    cosine_mid = get_alpha_schedule(5, 10, 0.01, 0.3, schedule_type="cosine")
    exp_mid = get_alpha_schedule(5, 10, 0.01, 0.3, schedule_type="exponential")
    assert 0.01 <= linear_mid <= 0.3
    assert 0.01 <= cosine_mid <= 0.3
    assert 0.01 <= exp_mid <= 0.3
    assert get_alpha_schedule(0, 1, 0.01, 0.3) == pytest.approx(0.3)


def test_alpha_schedule_unsafe_configs_raise():
    with pytest.raises(ValueError):
        get_alpha_schedule(0, 10, -0.1, 0.3)
    with pytest.raises(ValueError):
        get_alpha_schedule(0, 10, 0.1, 1.2)


def test_model_registry_aliases():
    assert resolve_model_spec("phi3").name.endswith("phi-3-mini-4k-instruct")
    assert resolve_model_spec("llama-3").name.startswith("meta-llama")
    assert resolve_model_spec("mistral").name.startswith("mistralai")


def test_rigor_reproducible_and_identical_distributions():
    vals = [1, 1, 1, 1, 1]
    a = bootstrap_ci(vals, num_bootstrap=200, seed=7)
    b = bootstrap_ci(vals, num_bootstrap=200, seed=7)
    assert a == b

    same = paired_bootstrap_test([1, 0, 1], [1, 0, 1], num_bootstrap=200, seed=1)
    assert same.effect_size == 0
    assert same.ci_low <= same.ci_high


def test_inference_engine_with_pruning_system_batch_len_mix():
    class DummySystem:
        def generate(self, prompt, max_new_tokens=128, use_pruning=True, budget_keep_ratio=None):
            return f"{prompt}:{max_new_tokens}:{use_pruning}:{budget_keep_ratio}"

    eng = InferenceEngine(model_name="dummy", pruning_system=DummySystem())
    out = eng.generate("a b c", config=GenerationConfig(max_new_tokens=5), budget_keep_ratio=0.9)
    assert out.endswith(":5:True:0.9")


def test_precision_cpu_fallback_fp16_and_bf16():
    model = nn.Linear(4, 1)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    def loss_fn(x, y):
        return torch.nn.functional.mse_loss(model(x), y)

    x = torch.randn(2, 4)
    y = torch.randn(2, 1)

    for p in ["fp16", "bf16", "fp32"]:
        trainer = MixedPrecisionTrainer(model, optim, PrecisionConfig(p))
        val = trainer.train_step(loss_fn, x, y)
        assert np.isfinite(val)


def test_temperature_guards_and_warning():
    with pytest.raises(ValueError):
        MicrogliaAgent(hidden_dim=16, num_heads=4, temperature=0.0)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        a = MicrogliaAgent(hidden_dim=16, num_heads=4, temperature=0.05)
        a.set_temperature(0.05)
        assert any("temperature < 0.1" in str(w.message) for w in rec)


def test_checkpoint_roundtrip_and_missing_agents_error(tmp_path):
    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(8, 8)
        def forward(self, hidden_states, output_attentions=False, **kwargs):
            attn = torch.softmax(torch.randn(hidden_states.shape[0], 2, hidden_states.shape[1], hidden_states.shape[1]), dim=-1)
            out = self.o_proj(hidden_states)
            return (out, attn) if output_attentions else out

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttention()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer()])
            self.config = nn.Module()
            self.config.num_attention_heads = 2
            self.config.pad_token_id = 0
            self.config.eos_token_id = 1

    sys = MicrogliaPruningSystem(model=FakeModel(), num_heads=2, hidden_dim=8, device="cpu")
    ckpt = tmp_path / "ckpt.pt"
    sys.save_checkpoint(str(ckpt))
    sys.load_checkpoint(str(ckpt), load_lora=False)

    bad = tmp_path / "bad.pt"
    torch.save({"x": 1}, bad)
    with pytest.raises(KeyError):
        sys.load_checkpoint(str(bad), load_lora=False)
