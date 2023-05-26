import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy import ndimage
from skimage.feature import peak_local_max




snake = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/snake.txt")
Im = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/flower.png")

print(snake.shape)

mask = skimage.draw.polygon2mask(Im.shape,snake)
mean_inside = np.sum(np.sum(mask*Im)) / np.sum(np.sum(mask))
mean_outside = np.sum(np.sum((1-mask)*Im)) / np.sum(np.sum(1-mask))


F_ext = (mean_inside-mean_outside)*(2*Im[int(snake[0,0]),int(snake[0,1])]-mean_inside-mean_outside)

print(F_ext)




