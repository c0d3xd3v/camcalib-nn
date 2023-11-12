import torch
import torch.nn as nn

class FocAndDisOut(nn.Module):
    def __init__(self, num_inputs=2048):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.fc = nn.Linear(num_inputs, 2)

    def forward(self, x):
        x = self.activation(x)
        x = self.fc(x)
        return x
