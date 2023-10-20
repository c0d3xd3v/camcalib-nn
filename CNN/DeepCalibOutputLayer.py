import math

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


# https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch
def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(_log_cosh(y_pred - y_true))
    

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)
        