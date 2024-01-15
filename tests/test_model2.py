# tests/test_model2.py
import torch
from pistachio.models.model import MyAwesomeModel

def test_model_architecture():
    # Instantiate your model
    model = MyAwesomeModel()

    # Assert that the model is an instance of nn.Module
    assert isinstance(model, torch.nn.Module), "The model should be an instance of torch.nn.Module"

    # Assert the existence of model components
    assert hasattr(model, 'flatten'), "The model should have a 'flatten' layer"
    assert hasattr(model, 'fc1'), "The model should have an 'fc1' layer"
    assert hasattr(model, 'relu'), "The model should have a 'relu' layer"
    assert hasattr(model, 'fc2'), "The model should have an 'fc2' layer"
    assert hasattr(model, 'fc3'), "The model should have an 'fc3' layer"

