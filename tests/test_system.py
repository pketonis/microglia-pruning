"""Integration tests for the complete pruning system."""

import torch
import torch.nn as nn
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system import MicrogliaPruningSystem

class FakeAttention(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        batch, seq, dim = hidden_states.shape
        # Fake attention weights
        attn_weights = torch.ones(batch, self.num_heads, seq, seq, device=hidden_states.device)
        # Fake output
        attn_output = self.o_proj(hidden_states)
        if output_attentions:
            return attn_output, attn_weights
        return attn_output

class FakeLayer(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.self_attn = FakeAttention(hidden_dim, num_heads)

class FakeModelInternal(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=64, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(hidden_dim, num_heads) for _ in range(num_layers)])

class FakeModel(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=64, num_heads=4):
        super().__init__()
        self.model = FakeModelInternal(num_layers, hidden_dim, num_heads)
        self.config = nn.Module()
        self.config.hidden_size = hidden_dim
        self.config.num_attention_heads = num_heads
        self.config.pad_token_id = 0
        self.config.eos_token_id = 2
        self.config._name_or_path = "fake-model"

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Fake forward pass
        # Just use some weights to have gradients
        x = torch.randn(input_ids.shape[0], input_ids.shape[1], 64, device=input_ids.device, requires_grad=True)
        for layer in self.model.layers:
            x = layer.self_attn(x, attention_mask=attention_mask)
            if isinstance(x, tuple):
                x = x[0]

        loss = x.sum()
        class Output:
            def __init__(self, loss):
                self.loss = loss
        return Output(loss)

class TestSystemIntegration:
    """Test suite for MicrogliaPruningSystem integration."""

    def test_system_initialization(self):
        """Test that the system initializes with a model and wraps it."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        system = MicrogliaPruningSystem(model=fake_model, num_heads=4, hidden_dim=32, device='cpu')

        # Check agents were initialized
        assert len(system.agents) == 2

        # Wrap layers
        system._wrap_attention_layers()
        assert system.wrapped

        # Check that layers were replaced
        layers = system.get_layers()
        for layer in layers:
            from src.pruned_attention import PrunedAttention
            assert isinstance(layer.self_attn, PrunedAttention)

    def test_extract_answer_robustness(self):
        """Test robust numerical answer extraction across common output formats."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        system = MicrogliaPruningSystem(model=fake_model, num_heads=4, hidden_dim=32, device='cpu')

        assert system._extract_answer("the answer is 42 students") == 42.0
        assert system._extract_answer("#### 100") == 100.0
        assert system._extract_answer("so there are 3.5 times as many, meaning 7 total") == 7.0
        assert system._extract_answer("no numbers here") is None
        assert system._extract_answer("$1,234.56 dollars") == 1234.56
        assert system._extract_answer(
            "Question: If 3+4?\nAnswer: let's think. 3 and 4 make 7.\nAnswer: 7"
        ) == 7.0

    def test_generate_toggles_pruning(self):
        """Test that generate method correctly toggles pruning state."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        system = MicrogliaPruningSystem(model=fake_model, num_heads=4, hidden_dim=32, device='cpu')
        system._wrap_attention_layers()

        # Mock tokenizer
        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                class BatchEncoding(dict):
                    def to(self, device):
                        return self
                return BatchEncoding({'input_ids': torch.zeros(1, 10, dtype=torch.long),
                                     'attention_mask': torch.ones(1, 10, dtype=torch.long)})
            def decode(self, tokens, **kwargs):
                return "Fake response"

        system.tokenizer = FakeTokenizer()

        # Mock model.generate
        def fake_generate(**kwargs):
            return torch.zeros(1, 20, dtype=torch.long)
        system.model.generate = fake_generate

        # Test with pruning enabled
        _ = system.generate("Test", use_pruning=True)
        assert system.pruning_enabled == True

        # Test with pruning disabled
        _ = system.generate("Test", use_pruning=False)
        assert system.pruning_enabled == False

    def test_train_mock(self, monkeypatch):
        """Test the training pipeline with mocked dataset."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        system = MicrogliaPruningSystem(model=fake_model, num_heads=4, hidden_dim=32, device='cpu')

        # Mock tokenizer
        class FakeTokenizer:
            def __init__(self):
                self.eos_token = "<|end|>"
                self.pad_token = "<|end|>"
            def __call__(self, text, **kwargs):
                batch_size = len(text) if isinstance(text, list) else 1
                class BatchEncoding(dict):
                    def to(self, device): return self
                return BatchEncoding({'input_ids': torch.zeros(batch_size, 10, dtype=torch.long),
                                     'attention_mask': torch.ones(batch_size, 10, dtype=torch.long)})
            def decode(self, tokens, **kwargs):
                return "Question: 1+1 Answer: 2"
        system.tokenizer = FakeTokenizer()

        # Mock load_dataset
        def mock_load_dataset(*args, **kwargs):
            class MockSplit:
                def select(self, indices): return self
                def __len__(self): return 10
                def __getitem__(self, idx): return {'question': '1+1', 'answer': '2'}
                def __iter__(self):
                    for i in range(2): yield {'question': '1+1', 'answer': '2'}
            return {'train': MockSplit(), 'test': MockSplit()}

        import src.system
        monkeypatch.setattr(src.system, "load_dataset", mock_load_dataset)

        # Run minimal training
        system.train(num_epochs=1, max_steps_per_epoch=2, batch_size=1, use_lora=False, use_budget=True)
        assert len(system.training_history) > 0

    def test_evaluate_mock(self, monkeypatch):
        """Test the evaluation pipeline with mocked dataset."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        system = MicrogliaPruningSystem(model=fake_model, num_heads=4, hidden_dim=32, device='cpu')
        system._wrap_attention_layers()

        # Mock tokenizer
        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                class BatchEncoding(dict):
                    def to(self, device): return self
                return BatchEncoding({'input_ids': torch.zeros(1, 10, dtype=torch.long),
                                     'attention_mask': torch.ones(1, 10, dtype=torch.long)})
            def decode(self, tokens, **kwargs):
                return "#### 2"
        system.tokenizer = FakeTokenizer()

        # Mock model.generate
        def fake_generate(*args, **kwargs):
            return torch.zeros(1, 20, dtype=torch.long)
        system.model.generate = fake_generate

        # Mock load_dataset
        def mock_load_dataset(*args, **kwargs):
            class MockSplit:
                def select(self, indices): return self
                def __len__(self): return 2
                def __iter__(self):
                    for i in range(2): yield {'question': '1+1', 'answer': '2'}
            return MockSplit()

        import src.system
        monkeypatch.setattr(src.system, "load_dataset", mock_load_dataset)

        results = system.evaluate(max_samples=2)
        assert 'accuracy' in results
        assert results['total'] == 2

    def test_checkpoint_save_load_robustness(self, tmp_path, monkeypatch):
        """Test saving and loading checkpoints with auto-LoRA and re-init."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        system = MicrogliaPruningSystem(model=fake_model, num_heads=4, hidden_dim=32, device='cpu')

        system._apply_lora()
        system._wrap_attention_layers()

        ckpt_path = str(tmp_path / "test_ckpt.pt")
        system.save_checkpoint(ckpt_path)

        # New system with mismatching hidden_dim
        new_fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        new_system = MicrogliaPruningSystem(model=new_fake_model, num_heads=4, hidden_dim=64, device='cpu')

        # Test loading with budget
        system._set_budget_keep_ratio(0.5)
        system.save_checkpoint(ckpt_path)

        # Should auto-apply LoRA and re-init agents
        new_system.load_checkpoint(ckpt_path, load_lora=False)
        assert new_system.lora_applied
        assert new_system.agents[0].fc1.out_features == 32
        assert new_system.last_budget == 0.5

    def test_system_initialization_with_string_model(self, monkeypatch):
        """Test system initialization with a model name string."""
        class MockConfig:
            def __init__(self):
                self.num_attention_heads = 4
                self.rope_scaling = None
                self._name_or_path = "mock-model"

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([FakeLayer()])
            def to(self, *args, **kwargs): return self

        def mock_from_pretrained(*args, **kwargs):
            return MockModel()

        class MockTokenizer:
            def __init__(self):
                self.pad_token = None
                self.eos_token = "<|end|>"
                self.eos_token_id = 2
            def convert_tokens_to_ids(self, text): return 2

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **k: MockConfig())
        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", mock_from_pretrained)
        monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: MockTokenizer())

        system = MicrogliaPruningSystem(model="gpt2", num_heads=4, device='cpu')
        assert system.num_heads == 4
        assert len(system.agents) == 1

    def test_system_initialization_invalid_temp(self):
        """Test system initialization with invalid temperature."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            MicrogliaPruningSystem(model=fake_model, temperature=-1.0)

    def test_hard_prune_toggling(self):
        """Test that set_hard_prune toggles hard_prune on all layers."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        system = MicrogliaPruningSystem(model=fake_model, num_heads=4, hidden_dim=32, device='cpu')
        system._wrap_attention_layers()

        system.set_hard_prune(True)
        layers = system.get_layers()
        for layer in layers:
            assert layer.self_attn.hard_prune == True

        system.set_hard_prune(False)
        layers = system.get_layers()
        for layer in layers:
            assert layer.self_attn.hard_prune == False
