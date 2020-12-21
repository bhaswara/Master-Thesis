#This is used to measure image quality using brisque

import cv2
import numpy as np
import skimage
from skimage import measure
import os
import imquality.brisque as brisque

path1 = 'C:/Users/Irfan/Documents/01_thesis/ssim_files_FRGC/dim128/01-rwae_samples/'


img_base = []

for root, dirs, files in os.walk(path1):
    dirs.sort(key=int)

    if files == []:
        continue

    for filee in files:
        full_path = os.path.join(root, filee)
        #print(full_path)
        img = cv2.resize(cv2.imread(full_path), (64,64))
        img = skimage.color.rgb2gray(img)
        img_base.append(img)

img_base = np.array(img_base)

score = 0

for i in range(len(img_base)):
    calc = brisque.score(img_base[i])
    score += calc

print(score)
print(score / len(img_base))
