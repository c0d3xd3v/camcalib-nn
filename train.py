# export set PYTHONPATH=/home/kai/Development/densnet-pytorch/DeepCalib/dataset/:/home/kai/Development/densnet-pytorch/DeepCalib/network_training/Regression/Single_net/
import os, glob
import torch.optim as optim

from torchvision.models import inception_v3, Inception_V3_Weights
from DataSetGeneration.CustomImageDataset import *
from CNN.DeepCalibOutputLayer import *


output_dir = "continouse_dataset/"
labels_file = output_dir + "labels.csv"
img_dir = output_dir

inceptionV3 = torch.hub.load('pytorch/vision:v0.10.0',
                             'inception_v3',
                              weights=Inception_V3_Weights.IMAGENET1K_V1)
inceptionV3.fc = FocAndDisOut()
inceptionV3.aux_logits = False
#inceptionV3 = torch.load("deepcalib1.pt")

train_dataloader = loadDeepCaliDataNormalized(labels_file, img_dir) #loadDeepCaliData(labels_file, img_dir)

loss_fn = LogCoshLoss()
inceptionV3.train()

LR = 0.0001
for epoch, (train_feature, train_label) in enumerate(train_dataloader):
    print("epoch : " + str(epoch))
    print(str(epoch % 100) + ", " + str(LR))
    print(f"Feature batch shape: {train_feature.size()}")
    print(f"Labels batch shape: {train_label.size()}")
    optimizer = optim.Adam(inceptionV3.parameters(), lr=LR)
    predicted = inceptionV3(train_feature)
    loss = loss_fn(predicted[0], train_label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss : " + str(loss))
    torch.save(inceptionV3, "deepcalib1.pt")
