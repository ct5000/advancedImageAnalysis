import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy import ndimage
from skimage.feature import peak_local_max


surfaces = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/lines.txt")
Im = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/flower.png")
print(surfaces.shape)
max_cost = np.inf
idx = -1
for i in range(surfaces.shape[0]):
    max_delta_x = np.max(np.abs(surfaces[i,1:] - surfaces[i,:-1]))
    if max_delta_x > 3:
        pass
    else:
        cost = 0
        for j in range(surfaces.shape[1]):
            cost += Im[int(surfaces[i,j]),j]
        if cost < max_cost:
            max_cost = cost
            idx = i


print(idx)
