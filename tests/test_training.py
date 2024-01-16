import torch
from torch.utils.data import DataLoader
from pistachio.src.models.lightning_train import TransferLearningModel, PistachioDataModule


def test_transfer_learning_model_init_edge_cases():
    # Test with different values for num_classes
    model_1_class = TransferLearningModel(num_classes=1)
    model_5_classes = TransferLearningModel(num_classes=5)

