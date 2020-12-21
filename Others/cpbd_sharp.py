#This is used to measure the sharpness of the image using CPBD algorithm

import cv2
import numpy as np
from skimage import measure
import os
import cpbd

path1 = 'C:/Users/Irfan/Documents/01_thesis/ssim_files_FRGC/dim64/03-aae_samples/'

img_base = []

score = 0


for root, dirs, files in os.walk(path1):
    dirs.sort(key=int)

    if files == []:
        continue

    for filee in files:
        full_path = os.path.join(root, filee)
        #print(full_path)
        #img = cv2.resize(cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2GRAY), (64,64))
        img = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2GRAY)
        img = img[30:145,30:145]
        img_base.append(img)

img_base = np.array(img_base)
print(img_base.shape)


for i in range(len(img_base)):
    calc = cpbd.compute(img_base[i])
    score += calc

print(score)
print(score / len(img_base))