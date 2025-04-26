import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, hidden_units, num_classes):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x