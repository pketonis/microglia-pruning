import torch
import torch.nn as nn
import pytest
from src.utils import get_model_layers

def test_get_model_layers_standard():
    class SubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10)])

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SubModel()

    model = MockModel()
    layers = get_model_layers(model)
    assert len(layers) == 1

def test_get_model_layers_fallback_h():
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.ModuleList([nn.Linear(10, 10)])

    model = MockModel()
    layers = get_model_layers(model)
    assert len(layers) == 1

def test_get_model_layers_gpt2():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.ModuleList([nn.Linear(10, 10)])

    class MockGPT2(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Transformer()

    model = MockGPT2()
    layers = get_model_layers(model)
    assert len(layers) == 1

def test_get_model_layers_peft():
    class InternalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10)])

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = InternalModel()

    class MockPeft(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = Base()

    model = MockPeft()
    layers = get_model_layers(model)
    assert len(layers) == 1

def test_get_model_layers_alternative_peft():
    class InternalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10)])

    class MockPeft(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = InternalModel()

    model = MockPeft()
    layers = get_model_layers(model)
    assert len(layers) == 1

def test_get_model_layers_direct_h():
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.ModuleList([nn.Linear(10, 10)])

    model = MockModel()
    layers = get_model_layers(model)
    assert len(layers) == 1

def test_get_model_layers_transformer_layers():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10)])

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Transformer()

    model = MockModel()
    layers = get_model_layers(model)
    assert len(layers) == 1

def test_get_model_layers_fallback_layers():
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10)])

    model = MockModel()
    layers = get_model_layers(model)
    assert len(layers) == 1

def test_get_model_layers_fail():
    class UnknownModel(nn.Module):
        pass

    model = UnknownModel()
    with pytest.raises(AttributeError):
        get_model_layers(model)
