import os, sys, glob, time

import json

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

LR = 0.0125
accumulation_batch_size = 4
batch_size = int(sys.argv[1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = LogCoshLoss() #NCCLoss()
train_dataloader = loadDeepCaliData(labels_file, img_dir, batch_size)

inceptionV3 = loadInceptionV3Regression()
optimizer = optim.Adam(inceptionV3.parameters(), lr=LR, foreach=True, amsgrad=True)
inceptionV3,optimizer, epochStart, last_min_loss =  load_ckp(output_dir, inceptionV3, optimizer)

if torch.cuda.is_available():
    print("use cuda : yes")
else:
    print("use cuda : no")

inceptionV3.to(device)
inceptionV3.train()

loss_series = []
start = time.time()

for epoch, (train_feature, train_label) in enumerate(train_dataloader):

    train_feature, train_label = train_feature.to(device), train_label.to(device)
    optimizer.zero_grad()
    predicted = inceptionV3(train_feature)
    loss = loss_fn(predicted, train_label)
    loss.backward()
    optimizer.step()
    loss_series.append((epochStart + epoch, loss.item()))

    if loss.item() < last_min_loss:
        last_min_loss = loss.item()
        checkpoint = {
            'epoch': epoch + 1 + epochStart,
            'state_dict': inceptionV3.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_min_loss': last_min_loss
        }
        save_ckp(checkpoint, output_dir)
        print("saved epoch : " + str(epochStart + epoch) + ", loss : " + str(loss.item()))

    with open(output_dir + 'loss_history.csv', 'a') as file:
        ep = epochStart + epoch
        l = loss.item()
        file.write(f'{ep},{l}\n')
        file.close()

    end = time.time()
    diff = end - start
    diff_h = diff/3600.
    if(diff_h >= 5.):
        break
