import sys
import cv2

import numpy as np

from torchvision.transforms import ToTensor, Compose, ToPILImage

from tools.undistortion import undistSphIm, Params, cropImage, cropImageToRect, cropRect
from CNN.LoadCNN import loadInceptionV3Regression, load_eval

cap = cv2.VideoCapture('/home/kai/Downloads/Telegram Desktop/video_2023-11-09_16-37-33.mp4')
count = 0

model_path = sys.argv[1]
inceptionV3 = loadInceptionV3Regression()
inceptionV3 = load_eval(model_path, inceptionV3)
inceptionV3.eval()

transform=Compose([ToPILImage(), ToTensor()])

f_avg = 0
xi_avg = 0
n = 0

while cap.isOpened():
    ret,frame = cap.read()
    if ret != None:
        h, w, _ = frame.shape

        new_width = 299
        ratio = new_width / w # (or new_height / height)
        new_height = int(h * ratio)
        dimensions = (new_width, new_height)
        new_image = cv2.resize(frame, dimensions, interpolation=cv2.INTER_LINEAR)
        pred_img = cv2.resize(frame, (299, 299), interpolation=cv2.INTER_LINEAR)
        image = transform(new_image)
        pred_img = transform(pred_img)
        predicted = inceptionV3(pred_img.unsqueeze(0))

        numpy_image = cv2.merge([image[2].numpy(), image[1].numpy(), image[0].numpy()])
        Idis = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        f_avg = f_avg + predicted[0][0].item()
        xi_avg = xi_avg + predicted[0][1].item()
        n = n + 1
        f = f_avg/n
        xi = xi_avg/n

        ImH,ImW,_ = Idis.shape
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
        #img = cropImageToRect(undist_img, ImW, ImH)
        #x, y, w, h = cropRect(undist_img, ImW, ImH)
        #color = (255, 0, 0)
        #thickness = 2
        #image_with_rectangle = cv2.rectangle(undist_img.copy(), (x, y), (x+w, y+h), color, thickness)
        img = cropImage(undist_img)

        ImH, ImW, _ = Idis.shape
        maxS = np.max([ImW, ImH])
        image_size = 400
        img2 = cv2.resize(Idis, (int(ImW/maxS*image_size), int(ImH/maxS*image_size)))

        ImH, ImW, _ = img.shape
        maxS = np.max([ImW, ImH])
        img = cv2.resize(img, (int(ImW/maxS*image_size), int(ImH/maxS*image_size)))

        stackedimg = np.hstack((img2, img/255))

        print(predicted)

        cv2.imshow('window-name', stackedimg)
        #cv2.imwrite("frame%d.jpg" % count, frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows() # destroy all opened windows
