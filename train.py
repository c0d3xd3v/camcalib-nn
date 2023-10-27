import os, sys, glob, time

import torch
import torch.optim as optim
from torchvision.models import inception_v3, Inception_V3_Weights

from CNN.DeepCalibOutputLayer import LogCoshLoss
from CNN.LoadCNN import loadInceptionV3Regression
from DataSetGeneration.CustomImageDataset import *


output_dir = "continouse_dataset/"
labels_file = output_dir + "labels.csv"
img_dir = output_dir

inceptionV3 = loadInceptionV3Regression(output_dir)
if inceptionV3 is not None:
    LR = 2.31E-02
    accumulation_batch_size = 4
    batch_size = int(sys.argv[1])

    loss_fn = LogCoshLoss()
    train_dataloader = loadDeepCaliData(labels_file, img_dir, batch_size)
    optimizer = optim.SGD(inceptionV3.parameters(), lr=LR, momentum=0.75)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    inceptionV3.train()
    start = time.time()

    for epoch, (train_feature, train_label) in enumerate(train_dataloader):
        predicted = inceptionV3(train_feature)
        loss = loss_fn(predicted, train_label)
        loss.backward()

        if (epoch + 1) % accumulation_batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            torch.save(inceptionV3, output_dir + "deepcalib1.pt")
            torch.save(inceptionV3.state_dict(), output_dir + 'model_state.pth')

            print("epoch : " + str(epoch) + ", loss : " + str(loss.item()))

        end = time.time()
        diff = end - start
        diff_h = diff/3600.
        if(diff_h >= 5.):
            break
