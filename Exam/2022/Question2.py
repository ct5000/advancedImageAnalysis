import numpy as np
import skimage.io
from skimage.feature import peak_local_max


blobsLaplace = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/blobs_Laplace.tif")

local_min = peak_local_max(-blobsLaplace,min_distance = 1,threshold_abs=50)

print(blobsLaplace.shape)
print(local_min)





