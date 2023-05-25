import numpy as np
import skimage.io
import matplotlib.pyplot as plt


img1 = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/kidney_1.png")
img2 = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/kidney_2.png")

sift_desc1 = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/SIFT_1_descriptors.txt")
sift_desc2 = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/SIFT_2_descriptors.txt")

sift_coord1 = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/SIFT_1_coordinates.txt")
sift_corrd2 = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/SIFT_2_coordinates.txt")


print(sift_desc1.shape)

min_desc_idx = 0
min_dist = np.sum(np.power((sift_desc1[0,:]-sift_desc2[0,:]),2))


for i in range(1,sift_desc2.shape[0]):
    dist = np.sum(np.power((sift_desc1[0,:]-sift_desc2[i,:]),2))
    if dist < min_dist:
        min_dist = dist
        min_desc_idx = i


spat_dist = np.sqrt(np.sum(np.power(sift_coord1[0,:]-sift_corrd2[min_desc_idx,:],2)))

print(spat_dist)
