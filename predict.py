import os
import sys
import csv
import math

import torch
from   torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Compose, ToPILImage
import torch.optim as optim

from DataSetGeneration.CustomImageDataset import *
from CNN.LoadCNN import loadInceptionV3Regression, loadMobileNetRegression, save_ckp, load_ckp


output_dir = sys.argv[1]
data_dir = sys.argv[2]

labels_file = data_dir + "labels.csv"
img_dir = data_dir

LR = 0.001
dataset = CustomImageDataset(labels_file, img_dir,
                            transform=Compose([ToPILImage(), ToTensor()]),
                            target_transform=Compose([float]))

#inceptionV3 = loadInceptionV3Regression()
inceptionV3 = loadMobileNetRegression()
optimizer = optim.Adam(inceptionV3.parameters(), lr=LR, foreach=True, amsgrad=True)
inceptionV3,optimizer, epochStart, last_min_loss, _ =  load_ckp(output_dir + 'current_state.pt', inceptionV3, optimizer)
inceptionV3.eval()

train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
_sum = 0
_start_ = 0
file_path = output_dir + 'validate.csv'
if os.path.exists(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        _start_ = sum(1 for row in reader)
        print(_start_)
        csvfile.close()

with open(file_path, 'w') as csvfile:
    for epoch, (train_feature, train_label, path) in enumerate(train_dataloader):
        predicted = inceptionV3(train_feature)
        _sum = _sum + (predicted - train_label)**2
        print("----------------------------")
        print(train_label)
        print(predicted)
        print("----------------------------")
        #print(f'{path[0]}, {predicted[0][0]}, {predicted[0][1]}')
        csvfile.write(f'{path[0]}, {predicted[0][0]}, {predicted[0][1]}\n')
        _start_ = _start_ + 1
    csvfile.close()
    print(math.sqrt(torch.sum(_sum))/len(train_dataloader))
