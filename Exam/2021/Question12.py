import numpy as np
import skimage.io
import matplotlib.pyplot as plt




img = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2021/EXAM_DATA_2021/bony.png")

plt.figure()
plt.imshow(img)



mu = [130,190]
beta = 3000

img_bright = img > 140

print(np.sum(img_bright))


plt.show()