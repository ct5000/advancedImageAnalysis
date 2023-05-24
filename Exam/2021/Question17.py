import numpy as np


distances = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2021/EXAM_DATA_2021/layers.txt")


costs = np.zeros([5,1])

for i in range(distances.shape[1]):
    cost = np.zeros([5,1])
    for j in range(distances.shape[0]):
        cost[j,0] = np.sum(distances[j+1:,i]) + np.sum(20-distances[:j,i]) + (20-distances[j,i])
    costs += cost 
    

print(costs)




