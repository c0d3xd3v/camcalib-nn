import os
import time
import torch
from torch import nn
import torch.optim as optim
from torch_lr_finder import LRFinder

from CNN.DeepCalibOutputLayer import *
from torchvision.models import inception_v3, Inception_V3_Weights
from DataSetGeneration.CustomImageDataset import *

output_dir = "continouse_dataset/"
labels_file = output_dir + "labels.csv"
img_dir = output_dir
# Definieren Sie Ihr Modell und den Verlust (Loss)
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

criterion = LogCoshLoss()

# Definieren Sie den Optimizer mit einer vorläufigen Lernrate (Startwert)
optimizer = optim.Adam(inceptionV3.parameters(), lr=1e-7)

train_loader = loadDeepCaliData(labels_file, img_dir)

# Erstellen Sie den Lernratenfinder
lr_finder = LRFinder(inceptionV3, optimizer, criterion)

# Führen Sie den Lernratenfinder durch, indem Sie die Trainingsdaten durchlaufen
lr_finder.range_test(train_loader, end_lr=100, num_iter=100)

# Zeigen Sie die Ergebnisse des Lernratenfinders an
lr_finder.plot()

# Wählen Sie die optimale Lernrate auf Grundlage der Plot-Grafik
best_lr = lr_finder.history['lr'][lr_finder.history['loss'].idxmin()]
print(f"Optimale Lernrate: {best_lr}")

# Setzen Sie die optimale Lernrate für das eigentliche Training
optimizer.param_groups[0]['lr'] = best_lr

# Führen Sie nun das eigentliche Training mit der ausgewählten Lernrate durch
