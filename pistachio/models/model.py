# model.py
from torch import nn

class MyAwesomeModel(nn.Module):
    """My awesome model."""
    def __init__(self):
        super(MyAwesomeModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1080000, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

