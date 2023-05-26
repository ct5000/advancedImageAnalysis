import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy import ndimage
from skimage.feature import peak_local_max


def optimisedCalculateTranformation(P,Q,threshold):
    s_prim, R_prim,t_prim = calculateTranformation(P,Q)
    P_transformed = s_prim*R_prim@P + t_prim
    euclidean_distance = np.linalg.norm(Q-P_transformed,axis=0)
    Q_good = []
    P_good = []
    for i in range(Q.shape[1]):
        if euclidean_distance[i] < threshold:
            Q_good.append(Q[:,i])
            P_good.append(P[:,i])
    Q_good = np.array(Q_good).T
    P_good = np.array(P_good).T

    if Q_good.shape[0] < 1 or Q_good.shape[1] < 5:
        return optimisedCalculateTranformation(P,Q,threshold*2)
    else:
        s,R,t = calculateTranformation(P_good,Q_good)
        return s,R,t,threshold

def calculateTranformation(P,Q):

    mu_p = np.reshape(np.mean(P,axis=1),[2,1])
    mu_q = np.reshape(np.mean(Q,axis=1),[2,1])
    s = np.sum(np.linalg.norm(Q-mu_q))/np.sum(np.linalg.norm(P-mu_p))
    Cov = (Q-mu_q)@(P-mu_p).T
    U,_,Vt = np.linalg.svd(Cov)
    R_hat = U@Vt
    D = np.array([[1,0],[0,np.linalg.det(R_hat)]])
    R = R_hat@D
    t = mu_q - s*R@mu_p
    return s, R, t

sift_a = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/sift_a.txt")
sift_b = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/sift_b.txt")



print(sift_a.shape)
print(sift_b.shape)

match_desc = np.zeros([sift_a.shape[0]]).astype(int) # If negative no match

for i in range(sift_a.shape[0]):
    min_dist = 0
    min_dist_idx = 0
    sec_min_dist = 0
    dist1 = np.sum(np.power((sift_a[i,2:]-sift_b[0,2:]),2))
    dist2 = np.sum(np.power((sift_a[i,2:]-sift_b[1,2:]),2))
    if dist1 < dist2:
        min_dist = dist1
        sec_min_dist = dist2
    else:
        min_dist = dist2
        sec_min_dist = dist1
        min_dist_idx = 1
    for j in range(2,sift_b.shape[0]):
        dist = np.sum(np.power((sift_a[i,2:]-sift_b[j,2:]),2))
        if dist < min_dist:
            sec_min_dist = min_dist
            min_dist = dist
            min_dist_idx = j
        elif dist < sec_min_dist:
            sec_min_dist = dist
    if min_dist/sec_min_dist < 0.8:
        match_desc[i] = min_dist_idx
    else:
        match_desc[i] = -1


point1 = []
point2 = []

for i in range(sift_a.shape[0]):
    if match_desc[i] >= 0:
        point1.append(sift_a[i,:2])
        point2.append(sift_b[match_desc[i],:2])


point1 = np.array(point1)
point2 = np.array(point2)
print(point1.shape)


s, R, t,thres = optimisedCalculateTranformation(point1.T,point2.T,0.3)
print(t)
print(s)
print(R)
print(thres)

print(np.linalg.norm(t))
