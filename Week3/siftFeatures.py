import numpy as np
from scipy import linalg, interpolate, ndimage
import skimage.io
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max



def transformIm(im,theta,s):
    print(im.shape)
    rows,cols = im.shape
    rot_mat = cv2.getRotationMatrix2D(((cols-1.)/2.,(rows-1.)/2.),theta,s)
    r_im = cv2.warpAffine(im,rot_mat,(cols,rows)) 
    return r_im


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

    


imgbox = cv2.imread('Box4.bmp')
Image1 = cv2.cvtColor(imgbox, cv2.COLOR_BGR2GRAY)

Image2 = transformIm(Image1,30,1)


plt.figure()
plt.subplot(1,2,1)
plt.imshow(Image1)
plt.subplot(1,2,2)
plt.imshow(Image2)


sift = cv2.SIFT_create() # Sigma of the gaussian at octave 0
kp1, des1 = sift.detectAndCompute(Image1,None)
#sift = cv2.SIFT_create() # Sigma of the gaussian at octave 0
kp2, des2 = sift.detectAndCompute(Image2,None)
Image1_sift = cv2.drawKeypoints(Image1,kp1,Image1)
Image2_sift = cv2.drawKeypoints(Image2,kp2,Image2)


plt.figure()
plt.subplot(1,2,1)
plt.imshow(Image1_sift)
plt.subplot(1,2,2)
plt.imshow(Image2_sift)



bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
Lowe = 0.6
good = []
for m,n in matches:
    if m.distance < Lowe*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(Image1,kp1,Image2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure()
plt.imshow(img3)

points1 = np.array([kp1[good[i][0].queryIdx].pt for i in range(len(good))]).astype(int).T
points2 = np.array([kp2[good[i][0].trainIdx].pt for i in range(len(good))]).astype(int).T

s, R, t = calculateTranformation(points1,points2)

points1_trans = s*R@points1 + t


plt.figure()
plt.imshow(Image2_sift)
plt.plot(points1_trans[0],points1_trans[1],'bo')


s_optim,R_optim,t_optim,threshod = optimisedCalculateTranformation(points1,points2,20)

points1_trans_optim = s_optim*R_optim@points1 + t_optim
plt.figure()
plt.imshow(Image2_sift)
plt.plot(points1_trans_optim[0],points1_trans_optim[1],'bo')










plt.show()