import torch
from torch.utils.data import DataLoader
import pytest
from pistachio.src.models.lightning_train import TransferLearningModel, PistachioDataModule


# Mock for the PyTorch Lightning Trainer to avoid actual training
@pytest.fixture

def test_transfer_learning_model_construction():
    model = TransferLearningModel()

    # Check if the model is an instance of TransferLearningModel
    assert isinstance(model, TransferLearningModel)


def test_transfer_learning_model_configure_optimizers():
    model = TransferLearningModel()

    # Call the configure_optimizers method
    optimizer, scheduler = model.configure_optimizers()

    # Check if the optimizer and scheduler are instances of the expected classes
    assert isinstance(optimizer[0], torch.optim.Adam)
    assert isinstance(scheduler[0], torch.optim.lr_scheduler.MultiStepLR)


def test_transfer_learning_model_construction():
    model = TransferLearningModel()

    # Check if the model is an instance of TransferLearningModel
    assert isinstance(model, TransferLearningModel)

    # Check if the default values are set correctly
    assert model.backbone == "resnet50"
    assert not model.train_bn
    assert model.milestones == (2, 4)
    assert model.batch_size == 32
    assert model.lr == 1e-3
    assert model.lr_scheduler_gamma == 1e-1
    assert model.num_workers == 6

    # Check if the model has initialized the feature extractor and other components
    assert hasattr(model, 'feature_extractor')
    assert hasattr(model, 'fc')
    assert hasattr(model, 'loss_func')

    # Add more assertions based on your specific requirements

def test_transfer_learning_model_forward_pass():
    model = TransferLearningModel()

    # Mock input batch for forward pass
    batch_size = 32
    input_batch = torch.randn(batch_size, 3, 224, 224)

    # Call the forward method
    output = model.forward(input_batch)

    # Check if the output has the expected shape
    assert output.shape == (batch_size, 1)

    # Add more assertions based on your specific requirements

def test_transfer_learning_model_configure_optimizers():
    model = TransferLearningModel()

    # Call the configure_optimizers method
    optimizer, scheduler = model.configure_optimizers()

    # Check if the optimizer and scheduler are instances of the expected classes
    assert isinstance(optimizer[0], torch.optim.Adam)
    assert isinstance(scheduler[0], torch.optim.lr_scheduler.MultiStepLR)

    # Add more assertions based on your specific requirements

# Add more tests as needed




