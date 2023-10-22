# export set PYTHONPATH=/home/kai/Development/densnet-pytorch/DeepCalib/dataset/:/home/kai/Development/densnet-pytorch/DeepCalib/network_training/Regression/Single_net/
import os, glob
import torch.optim as optim

from torchvision.models import inception_v3, Inception_V3_Weights
from DataSetGeneration.CustomImageDataset import *
from CNN.DeepCalibOutputLayer import *


output_dir = "continouse_dataset/"
labels_file = output_dir + "labels.csv"
img_dir = output_dir

inceptionV3 = None
if os.path.isfile(output_dir + "deepcalib1.pt"):
    inceptionV3 = torch.load(output_dir + "deepcalib1.pt")
else:
    inceptionV3 = torch.hub.load('pytorch/vision:v0.10.0',
                                 'inception_v3',
                                  weights=Inception_V3_Weights.IMAGENET1K_V1)
    inceptionV3.fc = FocAndDisOut()
    inceptionV3.aux_logits = False

if inceptionV3 is not None:
    train_dataloader = loadDeepCaliData(labels_file, img_dir)

    loss_fn = LogCoshLoss()
    inceptionV3.train()

    print(f"Feature batch shape: {train_feature.size()}")
    print(f"Labels batch shape: {train_label.size()}")

    LR = 0.000001
    for epoch, (train_feature, train_label) in enumerate(train_dataloader):
        optimizer = optim.Adam(inceptionV3.parameters(), lr=LR)
        predicted = inceptionV3(train_feature)
        loss = loss_fn(predicted[0], train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch : " + str(epoch) + ", loss : " + str(loss))
        torch.save(inceptionV3, output_dir + "deepcalib1.pt")
