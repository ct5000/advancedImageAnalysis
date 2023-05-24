import numpy as np 
import math


points_p = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2021/EXAM_DATA_2021/points_p.txt")
points_q = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2021/EXAM_DATA_2021/points_q.txt")
# Shape [2,30]


theta = 140*math.pi/180
s = 1.7
t = np.array([[36,13]]).T
R = np.array([[math.cos(theta),-math.sin(theta)],
              [math.sin(theta),math.cos(theta)]])

q_cal = s*R@points_p + t

res = np.linalg.norm(points_q-q_cal,axis=0)

outlier = res>2

print(np.sum(outlier))
p_ = R.T@(points_q-t)/s

d = np.sqrt(np.sum((points_p - p_)**2,axis=0))
print(d)
print(f'Question 6: {np.sum(d>2)}')

