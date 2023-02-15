import numpy as np
from scipy import linalg, interpolate, ndimage
import skimage.io
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
import math




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


def optimisedCalculateTranformation(P,Q,threshold):
    s_prim, R_prim,t_prim = calculateTranformation(P,Q)
    P_transformed = s_prim*R_prim@P + t_prim
    euclidean_distance = np.linalg.norm(Q-P_transformed,axis=0)
    print(euclidean_distance.shape)
    Q_good = []
    P_good = []
    for i in range(Q.shape[1]):
        if euclidean_distance[i] < threshold:
            Q_good.append(Q[:,i])
            P_good.append(P[:,i])
    Q_good = np.array(Q_good).T
    P_good = np.array(P_good).T
    print(Q_good.shape)
    print(Q.shape)
    if Q_good.shape[0] < 5 and Q.shape[0] > 5:
        return optimisedCalculateTranformation(P,Q,threshold*2)
    else:
        s,R,t = calculateTranformation(P_good,Q_good)
        return s,R,t,threshold

def siftMatch(Im1,Im2,Lowe):
    sift = cv2.SIFT_create() # Sigma of the gaussian at octave 0
    kp1, des1 = sift.detectAndCompute(Image1,None)
    kp2, des2 = sift.detectAndCompute(Image2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < Lowe*n.distance:
            good.append([m])
    points1 = np.array([kp1[good[i][0].queryIdx].pt for i in range(len(good))]).astype(int).T
    points2 = np.array([kp2[good[i][0].trainIdx].pt for i in range(len(good))]).astype(int).T
    return points1,points2


Image1 = cv2.imread('quiz_image_1.png')
Image1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)


Image2 = cv2.imread('quiz_image_2.png')
Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)

points1, points2 = siftMatch(Image1,Image2,0.5)
print(points1.shape)

s,R,t,threshold = optimisedCalculateTranformation(points1,points2,20)






print(1/s)
print(np.arccos(R[0,0])%(2*np.pi))
print(np.arccos(R[0,0])*180/np.pi)
print(t)
print(threshold)
print(np.arcsin(R[1,0]))

print(math.atan2(R[0,0],math.sqrt(1-R[0,0]**2))%(2*math.pi)*(180/math.pi))

