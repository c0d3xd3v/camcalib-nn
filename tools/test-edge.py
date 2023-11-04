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
image = Image.open(r"/home/kai/Development/github/camcalib-nn/continouse_dataset/pano_0bd476aea59848b964b8506607a8c235_f_338_d_0.32846146695066664.jpg")


transform=Compose([ToEdgeImg(), ToTensor()])
img = transform(image)
display_image(img)

