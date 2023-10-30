import os, sys, glob, time, math

from CNN.DeepCalibOutputLayer import LogCoshLoss, NCCLoss
from CNN.LoadCNN import loadInceptionV3Regression, save_ckp, load_ckp
from DataSetGeneration.CustomImageDataset import *

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torchvision.models import inception_v3, Inception_V3_Weights
from torch_lr_finder import LRFinder

import json

def setLR(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return optimizer

def find_best_lr(model, optimizer, criterion, train_loader):
    lr_finder = LRFinder(model, optimizer, criterion)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)

    loss = lr_finder.history['loss']
    lr = lr_finder.history['lr']

    loss_gradients = np.gradient(loss)
    lgmin = np.min(loss_gradients)
    lgmin_index = list(loss_gradients).index(lgmin)
    best_lr = lr[lgmin_index]

    lr_finder.reset()
    lr_finder.plot()
    return best_lr

if __name__ == "__main__":
    output_dir = "continouse_dataset/"
    labels_file = output_dir + "labels.csv"
    img_dir = output_dir

    inceptionV3 = loadInceptionV3Regression()
    criterion = NCCLoss()
    optimizer = optim.Adam(inceptionV3.parameters(), foreach=True, amsgrad=True)
    train_loader = loadDeepCaliData(labels_file, img_dir, 4)
    inceptionV3, optimizer, epochStart =  load_ckp(output_dir, inceptionV3, optimizer)

    best_lr = find_best_lr(inceptionV3, optimizer, criterion, train_loader)
    print('{:.2e}'.format(best_lr))
