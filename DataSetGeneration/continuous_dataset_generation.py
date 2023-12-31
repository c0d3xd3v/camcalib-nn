import numpy as np
import csv
import cv2
import math as m
import time, glob
import DataSetGeneration.my_interpol as my_interpol
import random
import pdb
from numpy.lib.scimath import sqrt as csqrt

random.seed(9001)
np.random.seed(1)

def deg2rad(deg):
    return deg*m.pi/180

def getRotationMat(roll, pitch, yaw):

    rx = np.array([1., 0., 0., 0., np.cos(deg2rad(roll)), -np.sin(deg2rad(roll)), 0., np.sin(deg2rad(roll)), np.cos(deg2rad(roll))]).reshape((3, 3))
    ry = np.array([np.cos(deg2rad(pitch)), 0., np.sin(deg2rad(pitch)), 0., 1., 0., -np.sin(deg2rad(pitch)), 0., np.cos(deg2rad(pitch))]).reshape((3, 3))
    rz = np.array([np.cos(deg2rad(yaw)), -np.sin(deg2rad(yaw)), 0., np.sin(deg2rad(yaw)), np.cos(deg2rad(yaw)), 0., 0., 0., 1.]).reshape((3, 3))

    return np.matmul(rz, np.matmul(ry, rx))

def minfocal( u0,v0,xi,xref=1,yref=1):
    z = (1-xi*xi)*((xref-u0)*(xref-u0) + (yref-v0)*(yref-v0))
    fmin = np.sqrt(z) if(z >= 0.) else -1.
    return fmin * 1.0001

def diskradius(xi, f):
    z =-(f*f)/(1-xi*xi)
    return np.sqrt(z) if(z > 0.) else -1.

def generateProjection(f, xi, u0, v0, x_ref, y_ref, grid_x, grid_y, ImPano_W, ImPano_H):
    fmin = minfocal(u0, v0, xi, x_ref, y_ref)

    # 1. Projection on the camera plane

    X_Cam = np.divide(grid_x- u0, f)
    Y_Cam = np.divide(grid_y- v0, f)

    # 2. Projection on the sphere

    AuxVal = np.multiply(X_Cam, X_Cam) + np.multiply(Y_Cam, Y_Cam)

    alpha_cam = np.real(xi + csqrt(1 + np.multiply((1 - xi * xi), AuxVal)))

    alpha_div = AuxVal + 1

    alpha_cam_div = np.divide(alpha_cam, alpha_div)

    X_Sph = np.multiply(X_Cam, alpha_cam_div)
    Y_Sph = np.multiply(Y_Cam, alpha_cam_div)
    Z_Sph = alpha_cam_div - xi

    # # 3. Rotation of the sphere
    Rot = []
    Rot.append(((np.random.ranf() - 0.5) * 2) * 10) # roll
    Rot.append(((np.random.ranf() - 0.5) * 2) * 15) # pitch
    Rot.append(((np.random.ranf() - 0.5) * 2) * 180) # yaw

    r = np.matmul(getRotationMat(Rot[0], Rot[1], Rot[2]),
                np.matmul(getRotationMat(0, -90, 45), getRotationMat(0, 90, 90)))

    idx1 = np.array([[0], [0], [0]])
    idx2 = np.array([[1], [1], [1]])
    idx3 = np.array([[2], [2], [2]])
    elems1 = r[:, 0]
    elems2 = r[:, 1]
    elems3 = r[:, 2]

    x1 = elems1[0] * X_Sph + elems2[0] * Y_Sph + elems3[0] * Z_Sph
    y1 = elems1[1] * X_Sph + elems2[1] * Y_Sph + elems3[1] * Z_Sph
    z1 = elems1[2] * X_Sph + elems2[2] * Y_Sph + elems3[2] * Z_Sph

    X_Sph = x1
    Y_Sph = y1
    Z_Sph = z1

    # 4. cart 2 sph
    ntheta = np.arctan2(Y_Sph, X_Sph)
    nphi = np.arctan2(Z_Sph, np.sqrt(np.multiply(X_Sph, X_Sph) + np.multiply(Y_Sph, Y_Sph)))

    pi = m.pi

    # 5. Sphere to pano
    min_theta = -pi
    max_theta = pi
    min_phi = -pi / 2.
    max_phi = pi / 2.

    min_x = 0
    max_x = ImPano_W - 1.0
    min_y = 0
    max_y = ImPano_H - 1.0

    ## for x
    a = (max_theta - min_theta) / (max_x - min_x)
    b = max_theta - a * max_x  # from y=ax+b %% -a;
    nx = (1. / a)* (ntheta - b)

    ## for y
    a = (min_phi - max_phi) / (max_y - min_y)
    b = max_phi - a * min_y  # from y=ax+b %% -a;
    ny = (1. / a)* (nphi - b)

    return nx, ny, fmin


def generateImageProjections(image360, ImPano_W, ImPano_H, image360_path,output_path, numImg):
        H=299
        W=299
        u0 = W / 2.
        v0 = H / 2.
        grid_x, grid_y = np.meshgrid(range(W), range(H))

        csv_file = open(output_path + 'labels.csv', 'a', newline='')

        for i in range(numImg):
            while True:
                x_ref = 1
                y_ref = 1
                
                skip = False
                f = random.randint(230, 450)
                xi = random.uniform(0.12, 1.0)
                nx, ny, fmin = generateProjection(f, xi, u0, v0, x_ref, y_ref, grid_x, grid_y,ImPano_W, ImPano_H)

                # 6. Final step interpolation and mapping
                im = np.array(my_interpol.interp2linear(image360, nx, ny), dtype=np.uint8)

                if False: #f < fmin:
                    r = diskradius(xi, f)
                    if(r < 0.):
                        skip = True
                    DIM = im.shape
                    ci = (np.round(DIM[0]/2), np.round(DIM[1]/2))
                    xx, yy = np.meshgrid(range(DIM[0])-ci[0], range(DIM[1])-ci[1])
                    mask = np.double((np.multiply(xx,xx)+np.multiply(yy,yy))<r*r)
                    mask_3channel = np.stack([mask,mask,mask],axis=-1)
                    im = np.array(np.multiply(im, mask_3channel),dtype=np.uint8)
                if(skip == False):
                    break
        
            name = image360_path.split('/')[-1]
            name_list = name.split('.')

            if skip == False:
                writer = csv.writer(csv_file)
                file_path = output_path + name_list[0] +'_f_'+str(f)+'_d_'+str(xi)+ '.' +name_list[-1]
                writer.writerow([file_path, f, xi])
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(file_path, im)
        csv_file.close()

# def ceil_power_of_10(n):
#    exp = log(n, 10)
#    exp = ceil(exp)
#    return 10**(exp-1)

def generateSingleImageProjections(in_360_image_path, output_path, numImg):
        file_name = in_360_image_path.split('/')[-1]
        file_path = in_360_image_path.replace(file_name, '')

        image360 = cv2.imread(in_360_image_path)
        ImPano_W = np.shape(image360)[1]
        ImPano_H = np.shape(image360)[0]

        generateImageProjections(image360, ImPano_W, ImPano_H, in_360_image_path, output_path, numImg)
