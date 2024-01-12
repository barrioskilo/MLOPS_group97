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
        # Check for the expected input shape
        if x.ndim != 4:
            raise ValueError('Expected input to be a 4D tensor')
        if x.shape[1] != 3 or x.shape[2] != 600 or x.shape[3] != 600:
            raise ValueError('Expected each sample to have shape [3, 600, 600]')

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

