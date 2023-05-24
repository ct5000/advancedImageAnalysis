import numpy as np


distances = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2021/EXAM_DATA_2021/distances.txt")
labels = np.loadtxt("Exam/2021/EXAM_DATA_2021/labels.txt")

l1 = 0
l2 = 0
for i in range(distances.shape[1]):
    if labels[i] == 1:
        l1 += 1
    else:
        l2 += 1


total = l1+l2
print(l1/total)







