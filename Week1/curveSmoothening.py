import numpy as np
from scipy import linalg, interpolate, ndimage
import skimage.io
import matplotlib.pyplot as plt
import cv2


dino = np.loadtxt("curves/dino.txt")
dino_noise = np.loadtxt("curves/dino_noisy.txt")

hand = np.loadtxt("curves/hand.txt")
hand_noise = np.loadtxt("curves/hand_noisy.txt")
print(type(dino))
'''
print(dino[0,:])

plt.figure(1)
plt.subplot(121)
plt.scatter(dino[:,0],dino[:,1])
plt.subplot(122)
plt.scatter(dino_noise[:,0],dino_noise[:,1])
'''

def curveSmooth(curve,lamb, steps):
    N = curve.shape[0]
    L = np.diag(-2*np.ones(N))+np.diag(np.ones(N-1),k=-1)+np.diag(np.ones(N-1),k=1)
    L[0,N-1] = 1
    L[N-1,0] = 1
    X = curve
    for i in range(steps):
        X = (np.eye(N)+lamb*L) @ X
    return X


def euclidianLength(p1,p2):
    length = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
    return length



new_dino = curveSmooth(dino_noise,0.19,1)

print("point 1: ", dino_noise[5])
print("point 2: ", dino_noise[6])
print("point 3: ", dino_noise[7])
print("Smooth point: ", new_dino[6])

print(dino_noise.shape)
print(new_dino.shape)

curveLength = euclidianLength(new_dino[0,:],new_dino[new_dino.shape[0]-1,:])

for i in range(1,new_dino.shape[0]):
    curveLength += euclidianLength(new_dino[i-1,:],new_dino[i,:])

print("quiz")
print(curveLength)

'''
plt.figure(2)
plt.subplot(121)
plt.scatter(dino_noise[:,0],dino_noise[:,1])
plt.subplot(122)
plt.scatter(new_dino[:,0],new_dino[:,1])

'''
def curveSmoothImplicit(curve,lamb):
    N = curve.shape[0]
    L = np.diag(-2*np.ones(N))+np.diag(np.ones(N-1),k=-1)+np.diag(np.ones(N-1),k=1)
    L[0,N-1] = 1
    L[N-1,0] = 1
    X = linalg.inv((np.eye(N)-lamb*L)) @ curve
    return X

'''
new_dino2 = curveSmoothImplicit(dino_noise,5)
plt.figure(3)
plt.subplot(121)
plt.scatter(dino_noise[:,0],dino_noise[:,1])
plt.subplot(122)
plt.scatter(new_dino2[:,0],new_dino2[:,1])
'''


def curveSmoothImplicit(curve,alpha, beta):
    N = curve.shape[0]
    A = np.diag(-2*np.ones(N))+np.diag(np.ones(N-1),k=-1)+np.diag(np.ones(N-1),k=1)
    A[0,N-1] = 1
    A[N-1,0] = 1
    B = np.diag(-6*np.ones(N))+np.diag(4*np.ones(N-1),k=-1)+np.diag(4*np.ones(N-1),k=1)+np.diag(-np.ones(N-2),k=-2)+np.diag(-np.ones(N-2),k=2)
    B[0,N-2] = -1
    B[N-2,0] = -1
    B[N-1,1] = -1
    B[1,N-1] = -1
    B[0,N-1] = 4
    B[N-1,0] = 4
    X = linalg.inv((np.eye(N)-alpha*A-beta*B)) @ curve
    return X

new_dino3 = curveSmoothImplicit(dino_noise,5,3)
plt.figure(4)
plt.subplot(121)
plt.scatter(dino_noise[:,0],dino_noise[:,1])
plt.subplot(122)
plt.scatter(new_dino3[:,0],new_dino3[:,1])


new_hand = curveSmoothImplicit(hand_noise,5,3)
plt.figure(5)
plt.subplot(121)
plt.scatter(hand_noise[:,0],hand_noise[:,1])
plt.subplot(122)
plt.scatter(new_hand[:,0],new_hand[:,1])






plt.show()