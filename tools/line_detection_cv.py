import os
import glob
import cv2
import numpy as np


ordner_pfad = "data/continouse_dataset/"
files = glob.glob(os.path.join(ordner_pfad, "*.jpg"))

def processImage(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #_,thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # RGB for matplotlib, BGR for imshow() !
    rgb *= np.array((0,0,10),np.uint8)
    out = np.bitwise_or(input_image, rgb)
    return out


for file in files:
    input_image = cv2.imread(file)
    gray_image = processImage(input_image)
    cv2.imshow("input", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
