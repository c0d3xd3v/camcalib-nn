import os, sys
import glob
import cv2
import numpy as np


ordner_pfad = sys.argv[1]
files = glob.glob(os.path.join(ordner_pfad, "*.jpg"))

def processImage(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #_,thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray_image,10,254,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi/180, 50, minLineLength, maxLineGap)
    rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # RGB for matplotlib, BGR for imshow() !
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
            pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
            cv2.polylines(rgb, [pts], True, (0,255,0))
    rgb *= np.array((0,0,10),np.uint8)
    #out = np.bitwise_or(input_image, rgb)
    return rgb


for file in files:
    input_image = cv2.imread(file)
    gray_image = processImage(input_image)
    cv2.imshow("input", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
