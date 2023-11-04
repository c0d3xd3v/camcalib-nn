import math

import torch
import torch.nn as nn

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


class NCCLoss(nn.Module):
    def __init__(self):
        super(NCCLoss, self).__init__()

    def forward(self, X, Y):
        mean1 = torch.mean(X)
        mean2 = torch.mean(Y)
        cross_corr = torch.sum((X - mean1) * (Y - mean2))
        std1 = torch.sqrt(torch.sum((X - mean1) ** 2))
        std2 = torch.sqrt(torch.sum((Y - mean2) ** 2))
        ncc_loss = 1 - (cross_corr / (std1 * std2))
        return ncc_loss
