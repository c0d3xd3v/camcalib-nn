import torch
import torch.nn as nn

class FocAndDisOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.activation(x)
        x = self.fc(x)
        return x
