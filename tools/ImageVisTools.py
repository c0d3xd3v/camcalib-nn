# export set PYTHONPATH=/home/kai/Development/densnet-pytorch/DeepCalib/dataset/:/home/kai/Development/densnet-pytorch/DeepCalib/network_training/Regression/Single_net/

import os
import cv2
import numpy as np

def displayImageTensorCV(img, title="displayImageTensorCV"):
    numpy_image = img.numpy()

    cv2_image = np.transpose(numpy_image, (1, 2, 0))
    cv2.imshow(title, cv2_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import CustomImageDataset
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import ToTensor, Compose, ToPILImage

    output_dir = "continouse_dataset/"
    labels_file = output_dir + "labels.csv"
    img_dir = output_dir

    dataset = CustomImageDataset.CustomImageDataset(
                labels_file, img_dir, 
                transform=Compose([ToPILImage(), ToTensor()]), 
                target_transform=Compose([float])
              )
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    train_feature, train_label = next(iter(train_dataloader))

    print(train_feature[0].size())

    displayImageTensorCV(train_feature[0])