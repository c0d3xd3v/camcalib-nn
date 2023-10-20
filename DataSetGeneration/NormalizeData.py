# This Python file uses the following encoding: utf-8
import os, glob

import matplotlib.pyplot as plt

import torch.optim as optim
from torchvision.models import inception_v3, Inception_V3_Weights

from DataSetGeneration.CustomImageDataset import *


def display_image(images):
  images_np = images.numpy()
  img_plt = images_np.transpose(0,2,3,1)
  # display 5th image from dataset
  plt.imshow(img_plt[0])
  plt.show()


if __name__ == "__main__":
    output_dir = "/home/kai/Development/densnet-pytorch/continouse_dataset/"
    labels_file = output_dir + "labels.csv"
    img_dir = output_dir

    train_dataloader = loadDeepCaliDataNormalized(labels_file, img_dir)

    images_normal, labels = next(iter(train_dataloader))

    display_image(images_normal)

