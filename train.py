import os, sys, glob, time

import torch
import torch.optim as optim
from torchvision.models import inception_v3, Inception_V3_Weights

from CNN.DeepCalibOutputLayer import LogCoshLoss, NCCLoss
from CNN.LoadCNN import loadInceptionV3Regression, save_ckp, load_ckp
from DataSetGeneration.CustomImageDataset import *

from lr_finder import find_best_lr, setLR

output_dir = "continouse_dataset/"
labels_file = output_dir + "labels.csv"
img_dir = output_dir

LR = 6.58E-03
accumulation_batch_size = 4
batch_size = int(sys.argv[1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = LogCoshLoss()
train_dataloader = loadDeepCaliData(labels_file, img_dir, batch_size)

inceptionV3 = loadInceptionV3Regression()
optimizer = optim.Adam(inceptionV3.parameters(), lr=LR)
inceptionV3,optimizer, epochStart =  load_ckp(output_dir, inceptionV3, optimizer)

if torch.cuda.is_available():
    print("use cuda : yes")
else:
    print("use cuda : no")

inceptionV3.to(device)
inceptionV3.train()


start = time.time()
for epoch, (train_feature, train_label) in enumerate(train_dataloader):

    train_feature, train_label = train_feature.to(device), train_label.to(device)

    predicted = inceptionV3(train_feature)
    loss = loss_fn(predicted, train_label)
    loss.backward()

    if (epoch + 1) % accumulation_batch_size == 0:
        optimizer.step()
        optimizer.zero_grad()

        checkpoint = {
            'epoch': epoch + 1 + epochStart,
            'state_dict': inceptionV3.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, output_dir)

        print("epoch : " + str(epochStart + epoch) + ", loss : " + str(loss.item()))

    end = time.time()
    diff = end - start
    diff_h = diff/3600.
    if(diff_h >= 5.):
        break
