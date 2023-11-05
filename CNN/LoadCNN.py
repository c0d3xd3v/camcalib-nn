# This Python file uses the following encoding: utf-8

import os
import torch
import torch.optim as optim
from torchvision.models import inception_v3
from CNN.DeepCalibOutputLayer import FocAndDisOut


def loadInceptionV3Regression():
    inceptionV3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', init_weights=False)
    inceptionV3.fc = FocAndDisOut()
    inceptionV3.aux_logits = False
    return inceptionV3


def save_ckp(state, checkpoint_dir):
    f_path = checkpoint_dir
    torch.save(state, f_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    iteration = 0
    last_min_loss = float('inf')
    if os.path.isfile(checkpoint_fpath):
        checkpoint = torch.load(checkpoint_fpath)
        iteration = checkpoint['iteration']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_min_loss = checkpoint['last_min_loss']
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
    return model, optimizer, iteration , last_min_loss, epoch
