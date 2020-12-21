#This is used to measure the quality of image based on PSNR

import cv2
import numpy as np
from math import log10, sqrt
import os
from skimage import measure

path1 = 'C:/Users/Irfan/Documents/01_thesis/ssim_files_FRGC/baseline_test/'
path2 = 'C:/Users/Irfan/Documents/01_thesis/ssim_files_FRGC/dim128/06-rw18_samples/'


img_base = []

img_test = []

for root, dirs, files in os.walk(path1):
    dirs.sort(key=int)

    if files == []:
        continue

    for filee in files:
        full_path = os.path.join(root, filee)
        #print(full_path)
        img = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2GRAY)
        img_base.append(img)

for root, dirs, files in os.walk(path2):
    dirs.sort(key=int)

    if files == []:
        continue

    for filee in files:
        full_path = os.path.join(root, filee)
        #print(full_path)
        img = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2GRAY)
        img_test.append(img)



img_base = np.array(img_base)
img_test = np.array(img_test)

def PSNR(original, compressed): 
    img1 = original.astype(np.float64) / 255.
    img2 = compressed.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)

    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    #max_pixel = 255.0
    psnr = 10 * log10(1. / sqrt(mse)) 
    return psnr

score = 0

for i in range(len(img_base)):
    psnr_calc = measure.compare_psnr(img_base[i], img_test[i], data_range=255)
    score += psnr_calc

print(score)
print(score / len(img_base))
