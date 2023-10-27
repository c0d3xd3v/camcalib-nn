# This Python file uses the following encoding: utf-8

import os, sys, glob, time
import torch
import torch.optim as optim
from torchvision.models import inception_v3, Inception_V3_Weights
from CNN.DeepCalibOutputLayer import FocAndDisOut

def loadInceptionV3Regression(output_dir):
    inceptionV3 = None
    if os.path.isfile(output_dir + "deepcalib1.pt"):
        last_modified = os.path.getmtime(output_dir + "deepcalib1.pt")
        formatted_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified))
        print(f'last modified : {formatted_date}')
        inceptionV3 = torch.load(output_dir + "deepcalib1.pt")
    else:
        inceptionV3 = torch.hub.load('pytorch/vision:v0.10.0',
                                     'inception_v3',
                                      weights=Inception_V3_Weights.IMAGENET1K_V1)
        inceptionV3.fc = FocAndDisOut()
        inceptionV3.aux_logits = False
    return inceptionV3
