# This Python file uses the following encoding: utf-8

import os, sys, glob, time
import torch
import torch.optim as optim
from torchvision.models import inception_v3, Inception_V3_Weights
from CNN.DeepCalibOutputLayer import FocAndDisOut

def loadInceptionV3Regression():

    inceptionV3 = torch.hub.load('pytorch/vision:v0.10.0',
                                 'inception_v3'
#                                  ,weights=Inception_V3_Weights.IMAGENET1K_V1
                                  )
    inceptionV3.fc = FocAndDisOut()
    inceptionV3.aux_logits = False

#    inceptionV3 = None
#    if os.path.isfile(output_dir + "deepcalib1.pt"):
#        last_modified = os.path.getmtime(output_dir + "deepcalib1.pt")
#        formatted_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified))
#        print(f'deepcalib1.pt loaded, last modified : {formatted_date}')
#        inceptionV3 = torch.load(output_dir + "deepcalib1.pt")
#    else:
#        inceptionV3 = torch.hub.load('pytorch/vision:v0.10.0',
#                                     'inception_v3',
#                                      weights=Inception_V3_Weights.IMAGENET1K_V1)
#        inceptionV3.fc = FocAndDisOut()
#        inceptionV3.aux_logits = False

#    if os.path.isfile(output_dir + 'model_state.pth'):
#        last_modified = os.path.getmtime(output_dir + 'model_state.pth')
#        formatted_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified))
#        print(f'model_state.pth loaded, last modified : {formatted_date}')
#        inceptionV3.load_state_dict(torch.load(output_dir + 'model_state.pth'))

    return inceptionV3

def save_ckp(state, checkpoint_dir):
    f_path = checkpoint_dir + 'checkpoint.pt'
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer):
    epoch = 0
    if os.path.isfile(checkpoint_fpath + "checkpoint.pt"):
        checkpoint = torch.load(checkpoint_fpath + 'checkpoint.pt')
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, epoch
