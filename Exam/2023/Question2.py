import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy import ndimage


zealand = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/zealand.txt")
print(zealand.shape)
bend_tot = np.linalg.norm(zealand[zealand.shape[0]-1,:]+zealand[1,:]-2*zealand[0,:])


for i in range(1,zealand.shape[0]-1):
    bend_tot += np.linalg.norm(zealand[i-1,:]+zealand[i+1,:]-2*zealand[i,:])

bend_tot += np.linalg.norm(zealand[zealand.shape[0]-2]+zealand[0,:]-2*zealand[zealand.shape[0]-1])

bend_avg = bend_tot/zealand.shape[0]

print(bend_avg)

