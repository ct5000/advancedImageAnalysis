import numpy as np
from scipy import linalg, interpolate, ndimage
import skimage.io
import matplotlib.pyplot as plt
import cv2
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




p = np.random.randint(0,100,[2,100])

t = np.array([[np.random.randint(-100,100)],[np.random.randint(-100,100)]])
theta = np.radians(np.random.randint(0,360))
R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
s = np.random.rand(1)*np.random.randint(0,5)

q_noise = np.random.normal(1,1,[2,100])
q = s*R@p + t + q_noise

plt.figure()
plt.plot(p[0,:],p[1,:],'bo')
plt.plot(q[0,:],q[1,:],'go')

s_est, R_est, t_est = calculateTranformation(p,q)

print('True')
print(s)
print(R)
print(t)



print('Estimates')
print(s_est)
print(R_est)
print(t_est)






plt.show()