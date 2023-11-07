import sys
import csv
import cv2
import numpy as np
from collections import namedtuple

import DataSetGeneration.my_interpol as my_interpol

Params = namedtuple("Params", "W H f xi")

def undistSphIm(Idis, Paramsd, Paramsund):

    # Kameraparameter für das ursprüngliche (verzerrte) Bild
    f_dist = Paramsd.f
    u0_dist = Paramsd.W / 2
    v0_dist = Paramsd.H / 2

    # Kameraparameter für das unverzerrte Bild
    f_undist = Paramsund.f
    u0_undist = Paramsund.W / 2
    v0_undist = Paramsund.H / 2

    xi = Paramsd.xi

    Image_und = np.zeros((Paramsund.H, Paramsund.W, 3))

    # 1. Projektion auf das Bild
    grid_x, grid_y = np.meshgrid(np.arange(1, Paramsund.W + 1), np.arange(1, Paramsund.H + 1))
    X_Cam = grid_x / f_undist - u0_undist / f_undist
    Y_Cam = grid_y / f_undist - v0_undist / f_undist
    Z_Cam = np.ones((Paramsund.H, Paramsund.W))

    # 2. Bild zu Kugelkartesische Koordinaten
    xi1 = 0
    alpha_cam = (xi1 * Z_Cam + np.sqrt(Z_Cam**2 + ((1 - xi1**2) * (X_Cam**2 + Y_Cam**2)))) / (X_Cam**2 + Y_Cam**2 + Z_Cam**2)
    X_Sph = X_Cam * alpha_cam
    Y_Sph = Y_Cam * alpha_cam
    Z_Sph = Z_Cam * alpha_cam - xi1

    # 3. Rückprojektion auf das verzerrte Bild
    den = xi * (np.sqrt(X_Sph**2 + Y_Sph**2 + Z_Sph**2)) + Z_Sph
    X_d = ((X_Sph * f_dist) / den) + u0_dist
    Y_d = ((Y_Sph * f_dist) / den) + v0_dist

    # 4. Endgültige Schritt: Interpolation und Zuordnung
    img = Idis.astype(np.float32)
    x_map = X_d.astype(np.float32)
    y_map = Y_d.astype(np.float32)
    for c in range(3):
        Image_und[:, :, c] = cv2.remap(img[:, :, c], x_map, y_map, interpolation=cv2.INTER_AREA)
    return Image_und

def cropImage(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return crop


output_dir = sys.argv[1]
with open(output_dir, 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        dist = 0.0
        print(row[0])
        f = float(row[1])
        xi = float(row[2])
        Idis = cv2.imread(row[0])

        dist = dist + xi
        ImH,ImW,_ = Idis.shape
        f_dist = f * (ImW/ImH) * (ImH/299)
        f = f + f_dist
        u0_dist = ImW/2
        v0_dist = ImH/2

        Paramsd = Params(int(u0_dist*2), int(v0_dist*2), f_dist, xi)
        Paramsund = Params(3*int(u0_dist*2), 3*int(v0_dist*2), f_dist,  0.0)

        undist_img = undistSphIm(Idis, Paramsd, Paramsund)

        undist_img = np.uint8(undist_img)
        img = cropImage(undist_img)
        img = cv2.resize(img, (ImW, ImH))

        stackedimg = np.hstack((Idis, img))

        cv2.imshow("Image with Contours", stackedimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f'f : {Paramsd.f} xi : {Paramsd.xi} dist : {Paramsd.W}')
    file.close()
