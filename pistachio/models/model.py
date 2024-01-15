# model.py
from torch import nn
from torch import Tensor

class MyAwesomeModel(nn.Module):
    """My awesome model."""
    def __init__(self):
        super(MyAwesomeModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1080000, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 classes for binary classification

    def forward(self, x: Tensor):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

