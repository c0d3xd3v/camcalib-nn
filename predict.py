# export set PYTHONPATH=/home/kai/Development/densnet-pytorch/DeepCalib/dataset/:/home/kai/Development/densnet-pytorch/DeepCalib/network_training/Regression/Single_net/

import os
import glob

import torch
from   torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Compose, ToPILImage

import CustomImageDataset


output_dir = "/home/kai/Development/densnet-pytorch/continouse_dataset/"
labels_file = output_dir + "labels.csv"
img_dir = output_dir

dataset = CustomImageDataset.CustomImageDataset(labels_file, img_dir, 
                            transform=Compose([ToPILImage(), ToTensor()]), 
                            target_transform=Compose([float]))

inceptionV3 = torch.load("deepcalib.pt")
inceptionV3.eval()

train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
train_feature, train_label = next(iter(train_dataloader))

predicted = inceptionV3(train_feature)
print(predicted)
print(train_label)
