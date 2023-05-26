import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy import ndimage
from skimage.feature import peak_local_max


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


p = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/points_p.txt")
q = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/points_q.txt")


s, R, t = calculateTranformation(p.T,q.T)

print("Scale")
print(s)
print("Trans")
print(t)
print("Rot")
print(R)
print(np.arctan2(R[1,0],R[0,0])*180/np.pi)

