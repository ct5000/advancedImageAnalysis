import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
from skimage import io
from scipy import ndimage

path = "C:/Users/andre/OneDrive/Skrivebord/Skole/Advanced Image Analysis/Project/advancedImageAnalysis/Project/"

image1 = io.imread(path + "optical_flow_data/composedIm_1.png").astype(np.float32)
image2 = io.imread(path + "optical_flow_data/composedIm_2.png").astype(np.float32)

im1norm = image1 / 255
im2norm = image2 / 255

c1 = (2, 6)
c2 = (5, 4)

k1 = [(c1[0]-1, c1[1]-1), (c1[0]-1, c1[1]), (c1[0]-1, c1[1]+1), (c1[0], c1[1]-1), (c1[0], c1[1]), (c1[0], c1[1]+1), (c1[0]+1, c1[1]-1), (c1[0]+1, c1[1]), (c1[0]+1, c1[1]+1)]
k2 = [(c2[0]-1, c2[1]-1), (c2[0]-1, c2[1]), (c2[0]-1, c2[1]+1), (c2[0], c2[1]-1), (c2[0], c2[1]), (c2[0], c2[1]+1), (c2[0]+1, c2[1]-1), (c2[0]+1, c2[1]), (c2[0]+1, c2[1]+1)]

kernel_x = np.array([[1/3, 1/3, 1/3]])

im1mean = ndimage.convolve(ndimage.convolve(im1norm,kernel_x),kernel_x.T)
im2mean = ndimage.convolve(ndimage.convolve(im2norm,kernel_x),kernel_x.T)

b1 = np.zeros(9)
b2 = np.zeros(9)

for i in range(9):
    b1[i] = (im1mean[k1[i]] - im2mean[k1[i]])
    b2[i] = im1norm[k2[i]] - im2norm[k2[i]]

g = np.array([[-1, 0, 1]])

convx = ndimage.convolve(im1norm,g)
convy = ndimage.convolve(im1norm,g.T)

A1 = np.zeros([9,2])
A2 = np.zeros([9,2])
for i in range(9):
    A1[i,1] = convy[k1[i]]
    A1[i,0] = convx[k1[i]]
    A2[i,1] = convy[k2[i]]
    A2[i,0] = convx[k2[i]]

u1 = np.linalg.pinv(A1)@b1
u2 = np.linalg.pinv(A2)@b2
#u1 = np.linalg.inv(A1.T@A1)@A1.T@b1