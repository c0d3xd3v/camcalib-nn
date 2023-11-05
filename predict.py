# export set PYTHONPATH=/home/kai/Development/densnet-pytorch/DeepCalib/dataset/:/home/kai/Development/densnet-pytorch/DeepCalib/network_training/Regression/Single_net/

import os
import sys

import torch
from   torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Compose, ToPILImage
import torch.optim as optim

from DataSetGeneration.CustomImageDataset import *
from CNN.LoadCNN import loadInceptionV3Regression, save_ckp, load_ckp


output_dir = sys.argv[1]
data_dir = sys.argv[2]

labels_file = data_dir + "labels.csv"
img_dir = data_dir

LR = 0.001
dataset = CustomImageDataset(labels_file, img_dir,
                            transform=Compose([ToPILImage(), ToTensor()]),
                            target_transform=Compose([float]))

inceptionV3 = loadInceptionV3Regression()
optimizer = optim.Adam(inceptionV3.parameters(), lr=LR, foreach=True, amsgrad=True)
inceptionV3,optimizer, epochStart, last_min_loss =  load_ckp(output_dir + 'current_state.pt', inceptionV3, optimizer)
inceptionV3.eval()

train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
(train_feature, train_label) = next(iter(train_dataloader))
predicted = inceptionV3(train_feature)
print("train_label : ")
print(train_label)
print("----------------------------")
print("predicted : ")
print(predicted)
print("----------------------------")
print(torch.sum(predicted - train_label)*0.5)

#sum = 0
#for epoch, (train_feature, train_label) in enumerate(train_dataloader):
#    predicted = inceptionV3(train_feature)
#    print("train_label : ")
#    print(train_label)
#    print("predicted : ")
#    print(predicted)
#    print(predicted - train_label)
#    sum = sum + predicted - train_label
#    print("---------------------------------------")
#print(sum/len(train_dataloader))
