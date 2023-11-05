import os, sys, time

import torch
import torch.optim as optim
from torchvision.models import inception_v3

from CNN.LoadCNN import loadInceptionV3Regression, save_ckp, load_ckp
from CNN.LossFunctions import selectLossFunction

from DataSetGeneration.CustomImageDataset import *


batch_size = int(sys.argv[1])
batch_accum = int(sys.argv[2])
num_epochs = int(sys.argv[3])
LR = float(sys.argv[4])
l2_lambda = float(sys.argv[5])
time_restrict=int(sys.argv[6])
output_dir = sys.argv[7]
data_dir = sys.argv[8]
loss_function_name = sys.argv[9]

labels_file = data_dir + "labels.csv"
img_dir = data_dir

if not os. path. exists(output_dir):
    os.mkdir(output_dir)

torch.set_num_threads(4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("use cuda : yes")
else:
    print("use cuda : no")

start = time.time()

loss_fn = selectLossFunction(loss_function_name)
train_dataloader = loadDeepCaliData(labels_file, img_dir, batch_size)
inceptionV3 = loadInceptionV3Regression()
inceptionV3.to(device)
inceptionV3.train()
optimizer = optim.Adam(inceptionV3.parameters(), lr=LR, weight_decay=l2_lambda, amsgrad=True)

count_epochs = 0
while True:
    inceptionV3, optimizer, iterationStart, last_min_loss, epoch = load_ckp(output_dir + 'current_state.pt', inceptionV3, optimizer)

    epoch = epoch + 1
    print("epoch      : " + str(epoch))
    print("current itr: " + str(iterationStart))
    print("iterations : " + str(len(train_dataloader)))

    for iteration, (train_feature, train_label) in enumerate(train_dataloader):
        # this steep is needed if a cuda device is available
        train_feature, train_label = train_feature.to(device), train_label.to(device)
        # gradient accumuation effectivly effects the batch size.
        if (iterationStart + iteration) % batch_accum == 0:
            optimizer.zero_grad()
        # for  training, pytorch needs same batch size for every batch.
        if train_feature.shape[0] != batch_size:
            break
        predicted = inceptionV3(train_feature)
        loss = loss_fn(predicted, train_label)
        loss.backward()
        optimizer.step()

        if loss.item() < last_min_loss:
            last_min_loss = loss.item()
            checkpoint = {
                'iteration': iteration + 1 + iterationStart,
                'state_dict': inceptionV3.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_min_loss': last_min_loss,
                'epoch' : epoch
            }
            save_ckp(checkpoint, output_dir + 'checkpoint.pt')
            with open(output_dir + 'checkpoint_history.csv', 'a') as file:
                ep = iterationStart + iteration
                l = loss.item()
                file.write(f'{ep},{l}\n')
                file.close()
            print("saved iteration : " + str(iterationStart + iteration) + ", loss : " + str(loss.item()))

        checkpoint = {
            'iteration': iteration + 1 + iterationStart,
            'state_dict': inceptionV3.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_min_loss': last_min_loss,
            'epoch' : epoch
        }
        save_ckp(checkpoint, output_dir + 'current_state.pt')

        with open(output_dir + 'loss_history.csv', 'a') as file:
            ep = iterationStart + iteration
            l = loss.item()
            file.write(f'{ep},{l}\n')
#            print(f'{ep},{l}')
            file.close()

        end = time.time()
        diff = end - start
        # convert to hours, thats the minimum scale we
        # are working with ;-) TODO: make it configurable from cmd
        diff_h = diff/3600.
        if(diff_h >= 5.0 and time_restrict==1):
            print("time restriciton activated. exit training.")
            break
    if count_epochs + 1 > num_epochs:
        break
    else:
        count_epochs = count_epochs + 1
