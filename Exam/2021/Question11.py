import numpy as np
import skimage.io
import matplotlib.pyplot as plt




img = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2021/EXAM_DATA_2021/circly.png")

plt.figure()
plt.imshow(img)
plt.show()


mu = [70,120,180]
beta = 100







