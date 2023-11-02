import os, sys, glob, time

import torch
import torch.optim as optim
from torchvision.models import inception_v3

from CNN.DeepCalibOutputLayer import LogCoshLoss, NCCLoss
from CNN.LoadCNN import loadInceptionV3Regression, save_ckp, load_ckp
from DataSetGeneration.CustomImageDataset import *

from lr_finder import find_best_lr, setLR

output_dir = "continouse_dataset/"
labels_file = output_dir + "labels.csv"
img_dir = output_dir

LR = 0.00005
l2_lambda = 0.01
batch_size = int(sys.argv[1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = LogCoshLoss()
#loss_fn = NCCLoss()
#loss_fn = torch.nn.MSELoss()
train_dataloader = loadDeepCaliData(labels_file, img_dir, batch_size)

inceptionV3 = loadInceptionV3Regression()
optimizer = optim.Adam(inceptionV3.parameters(), lr=LR, weight_decay=l2_lambda, amsgrad=True)
inceptionV3,optimizer, epochStart, last_min_loss =  load_ckp(output_dir + 'current_state.pt', inceptionV3, optimizer)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10)

if torch.cuda.is_available():
    print("use cuda : yes")
else:
    print("use cuda : no")

for param_group in optimizer.param_groups:
    param_group['lr'] = 1.0e-5

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
    #scheduler.step(loss)
    current_lr = optimizer.param_groups[0]['lr']

    loss_series.append((epochStart + epoch, loss.item()))
    if loss.item() < last_min_loss:
        last_min_loss = loss.item()
        checkpoint = {
            'epoch': epoch + 1 + epochStart,
            'state_dict': inceptionV3.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_min_loss': last_min_loss
        }
        save_ckp(checkpoint, output_dir + 'checkpoint.pt')
        print("saved epoch : " + str(epochStart + epoch) + ", loss : " + str(loss.item()))

    checkpoint = {
        'epoch': epoch + 1 + epochStart,
        'state_dict': inceptionV3.state_dict(),
        'optimizer': optimizer.state_dict(),
        'last_min_loss': last_min_loss
    }
    save_ckp(checkpoint, output_dir + 'current_state.pt')

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
