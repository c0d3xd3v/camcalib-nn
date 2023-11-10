import os, sys

import cv2
import numpy as np

import torch.optim as optim

from torchvision.io import read_image
from torchvision.transforms import ToTensor, Compose, ToPILImage

from CNN.LoadCNN import loadInceptionV3Regression, load_ckp

from tools.undistortion import undistSphIm, Params, cropImage, cropImageToRect

# data/val_dataset/logo_make_11_06_2023_388_f_375_d_0.8412931595250832.jpg
# data/val_dataset/logo_make_11_06_2023_388_f_387_d_0.6760518991470693.jpg

path = sys.argv[1]
model_path = sys.argv[2]

image_ = read_image(path)
transform=Compose([ToPILImage(), ToTensor()])
image = transform(image_)

inceptionV3 = loadInceptionV3Regression()
optimizer = optim.Adam(inceptionV3.parameters())
inceptionV3, _, _, _, _ = load_ckp(model_path, inceptionV3, optimizer)
inceptionV3.eval()
predicted = inceptionV3(image.unsqueeze(0))

print(predicted)

numpy_image = cv2.merge([image[0].numpy(), image[1].numpy(), image[2].numpy()])
Idis = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
f = predicted[0][0].item()
xi = predicted[0][1].item()

dist = 0.0
dist = dist + xi
ImH,ImW,_ = Idis.shape

# https://github.com/alexvbogdan/DeepCalib/issues/9
if ImH > ImW:
    f_scale = (ImH/ImW)*(ImH/299)
else:
    f_scale = (ImW/ImH)*(ImW/299)

f_dist = f * f_scale
f = f + f_dist
u0_dist = ImW/2
v0_dist = ImH/2

print(f'f : {f} xi : {xi}')


Paramsd = Params(int(u0_dist*2), int(v0_dist*2), f_dist, xi)
Paramsund = Params(3*int(u0_dist*2), 3*int(v0_dist*2), f_dist,  0.0)

undist_img = undistSphIm(Idis, Paramsd, Paramsund)
undist_img = np.uint8(undist_img*255)
img = cropImage(undist_img)
#img = cropImageToRect(undist_img, ImW, ImH)

ImH, ImW, _ = Idis.shape
maxS = np.min([ImW, ImH])
image_size = 400
img2 = cv2.resize(Idis, (int(ImW/maxS*image_size), int(ImH/maxS*image_size)))

ImH, ImW, _ = img.shape
maxS = np.min([ImW, ImH])
img = cv2.resize(img, (int(ImW/maxS*image_size), int(ImH/maxS*image_size)))

stackedimg = np.hstack((img2, img/255))

cv2.imshow("Image with Contours", stackedimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
