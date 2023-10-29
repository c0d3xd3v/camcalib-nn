import os, sys, glob, time

from CNN.DeepCalibOutputLayer import LogCoshLoss, NCCLoss
from CNN.LoadCNN import loadInceptionV3Regression
from DataSetGeneration.CustomImageDataset import *

import torch
import torch.optim as optim
from torchvision.models import inception_v3, Inception_V3_Weights
from torch_lr_finder import LRFinder


def find_best_lr(model, optimizer, criterion):
    lr_finder = LRFinder(model, optimizer, criterion)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)

    min_loss = min(lr_finder.history['loss'])
    idx_min_loss = lr_finder.history['loss'].index(min_loss)
    suggested_lr = lr_finder.history['lr'][idx_min_loss]

    print(min_loss)
    print(f"Beste Lernrate: {suggested_lr}")

    lr_finder.reset()
    lr_finder.plot()

    return suggested_lr

if __name__ == "__main__":
    output_dir = "continouse_dataset/"
    labels_file = output_dir + "labels.csv"
    img_dir = output_dir

    inceptionV3 = loadInceptionV3Regression(output_dir)
    criterion = NCCLoss() #LogCoshLoss()
    optimizer = optim.SGD(inceptionV3.parameters(), lr=1e-7, momentum=0.75)
    train_loader = loadDeepCaliData(labels_file, img_dir, 4)

    find_best_lr(inceptionV3, optimizer, criterion)
