import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy import ndimage
from skimage.feature import peak_local_max


img_org = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/cement.png").astype(np.float32)
blob_info = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/cement.txt")

plt.figure()
plt.imshow(img_org)


print(blob_info.shape)

dia = 0
count = 0

for i in range(200,blob_info.shape[0]):
    if blob_info[i,0] > 10:
        count += 1
        dia += np.sqrt(2*blob_info[i,3])*2

print(dia/count)



print("Try 2")
dia = 0
count = 0
for i in range(200,blob_info.shape[0]):
    if abs(blob_info[i,0]) > 10:
        if img_org[int(blob_info[i,1]),int(blob_info[i,2])] <150:
            count += 1
            dia += np.sqrt(2*blob_info[i,3])*2
print(dia/count)


plt.show()

