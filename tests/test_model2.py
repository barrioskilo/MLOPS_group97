# tests/test_model2.py
import torch
from pistachio.src.models.lightning_train import TransferLearningModel


def test_model_architecture():
    # Instantiate your model
    model = TransferLearningModel()

    # Assert that the model is an instance of nn.Module
    assert isinstance(model, torch.nn.Module), "The model should be an instance of torch.nn.Module"

    # Get the model architecture components
    feature_extractor = model.feature_extractor
    fc_layers = model.fc

    # Assert the existence of model components
    assert isinstance(feature_extractor, torch.nn.Sequential), "The model should have a 'feature_extractor' layer"
    assert isinstance(fc_layers, torch.nn.Sequential), "The model should have an 'fc' layer"
    
    # Assuming 'feature_extractor' has resnet layers
    assert isinstance(feature_extractor[0], torch.nn.Conv2d), "The feature extractor should have a Conv2d layer"
    
    # Assuming 'fc' has Linear and ReLU layers
    assert isinstance(fc_layers[0], torch.nn.Linear), "The fc layer should have a Linear layer"
    assert isinstance(fc_layers[1], torch.nn.ReLU), "The fc layer should have a ReLU layer"
    assert isinstance(fc_layers[2], torch.nn.Linear), "The fc layer should have another Linear layer"
    assert isinstance(fc_layers[3], torch.nn.Linear), "The fc layer should have another Linear layer"



