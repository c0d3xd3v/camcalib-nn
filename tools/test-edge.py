import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
from torchvision.transforms import ToTensor, Compose, ToPILImage

from DataSetGeneration.CustomImageDataset import *


def display_image(images):
    images_np = images.numpy()
    img_plt = images_np.transpose(2,1,0)
    plt.imshow(img_plt)
    plt.show()

# Opening the image (R prefixed to string
# in order to deal with '\' in paths)
image = Image.open(r"val_continouse_dataset/logo_make_11_06_2023_388_f_259_d_0.37045902112616813.jpg")


transform=Compose([ToTensor()])
img = transform(image)
display_image(img)

