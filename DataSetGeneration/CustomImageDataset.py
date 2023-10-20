import torch
import pandas as pd

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Compose, ToPILImage


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, delimiter=',')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]#os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        label2 = self.img_labels.iloc[idx, 2]
        #print(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            label2= self.target_transform(label2)
            tlabel = torch.tensor([label, label2])
        return image, tlabel


def loadDeepCaliData(labels_file, img_dir):
    dataset =  CustomImageDataset(labels_file, img_dir, 
                    transform=Compose([ToPILImage(), ToTensor()]), 
                    target_transform=Compose([float]))
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return train_dataloader

def batch_mean_and_sd(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)
    return mean,std


def loadDeepCaliDataNormalized(labels_file, img_dir):
    train_dataloader = loadDeepCaliData(labels_file, img_dir)

    mean, std = batch_mean_and_sd(train_dataloader)
    print("mean and std: \n", mean, std)

    normalized_dataset =  CustomImageDataset(labels_file, img_dir,
                    transform=Compose([
                        ToPILImage(),
                        ToTensor(),
                        torchvision.transforms.Normalize(mean = mean, std= std)]),
                        target_transform=Compose([float]))

    normalized_train_dataloader = DataLoader(normalized_dataset, batch_size=32, shuffle=True)

    return normalized_train_dataloader
