#This is used to measure the ssim of two images

import cv2
import numpy as np
from skimage import measure
import os

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

path1 = 'C:/Users/Irfan/Documents/01_thesis/ssim_files_FRGC/baseline_test/'
path2 = 'C:/Users/Irfan/Documents/01_thesis/ssim_files_FRGC/dim64/05-vae_samples/'


img_base = []

img_test = []

for root, dirs, files in os.walk(path1):
    dirs.sort(key=int)

    if files == []:
        continue

    for filee in files:
        full_path = os.path.join(root, filee)
        #print(full_path)
        img = cv2.imread(full_path)
        img_base.append(img)

for root, dirs, files in os.walk(path2):
    dirs.sort(key=int)

    if files == []:
        continue

    for filee in files:
        full_path = os.path.join(root, filee)
        #print(full_path)
        img = cv2.imread(full_path)
        img_test.append(img)



img_base = np.array(img_base)
img_test = np.array(img_test)

#score = measure.compare_ssim(img_test, img_base, gaussian_weights=True, multichannel=True)
#print(score)

score = 0

for i in range(len(img_base)):
    #ssim_calc = measure.compare_ssim(img_base[i], img_test[i],gaussian_weights=True, multichannel=True, data_range=255)
    ssim_calc = calculate_ssim(img_base[i], img_test[i])
    score += ssim_calc

print(score)
print(score / len(img_base))
