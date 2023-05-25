import numpy as np


configuration = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/configuration.txt")

V2 = 0

for i in range(configuration.shape[0]):
    for j in range(configuration.shape[1]):
        if i != 0:
            V2 += 10*abs(configuration[i,j]-configuration[i-1,j])
        if j != 0:
            V2 += 10*abs(configuration[i,j]-configuration[i,j-1])
        if i != configuration.shape[0]-1:
            V2 += 10*abs(configuration[i,j]-configuration[i+1,j])
        if j != configuration.shape[1]-1:
            V2 += 10*abs(configuration[i,j]-configuration[i,j+1])


print(V2/2)








