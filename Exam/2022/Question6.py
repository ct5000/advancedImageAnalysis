import numpy as np


C = np.array([[191955, -937044],
              [552183, 379358]])




U,_,Vt = np.linalg.svd(C)
R_hat = U@Vt
D = np.array([[1,0],[0,np.linalg.det(R_hat)]])
R = R_hat@D

print(R)
theta = np.arctan2(R[1,0],R[0,0])

print(theta*180/np.pi)

