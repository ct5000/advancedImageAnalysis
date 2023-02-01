import numpy as np
from scipy import linalg, interpolate, ndimage
import skimage.io
import matplotlib.pyplot as plt
import cv2



Cell1 = skimage.io.imread("fuel_cells/fuel_cell_1.tif")
Cell2 = skimage.io.imread("fuel_cells/fuel_cell_2.tif")
Cell3 = skimage.io.imread("fuel_cells/fuel_cell_3.tif")


plt.figure(1)
plt.subplot(131)
plt.imshow(Cell1)
plt.subplot(132)
plt.imshow(Cell2)
plt.subplot(133)
plt.imshow(Cell3)



def segmentationBoundaryLoop(segmentation):
    seg_length = 0
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            for k in range(-1,2):
                for l in range(-1,2):
                    if (i == 0 and k == -1):
                        pass
                    elif (j == 0 and l == -1):
                        pass
                    elif (i == segmentation.shape[0] - 1 and k == 1):
                        pass
                    elif (j == segmentation.shape[1] - 1 and l == 1):
                        pass
                    elif (l == 0 and k == 0):
                        pass
                    else:
                        if segmentation[i,j] == segmentation[i+k,j+l]:
                            pass
                        else:
                            seg_length += 1
    return seg_length




print("1: ", segmentationBoundaryLoop(Cell1))
print("2: ", segmentationBoundaryLoop(Cell2))
print("3: ", segmentationBoundaryLoop(Cell3))



























plt.show()


