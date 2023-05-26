import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy import ndimage
from skimage.feature import peak_local_max


labels = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/liver_assignment.png")
probabilities = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/dictionary_probabilities.txt")

l1 = 0
l2 = 0
l3 = 0




for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        label = labels[i,j] 
        prob = probabilities[label,:]
        max_idx = np.argmax(prob)
        if max_idx == 0:
            l1 += 1
        elif max_idx == 1:
            l2 += 1
        else:
            l3 += 1


print(l3/(l1+l2+l3))




























