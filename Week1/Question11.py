import numpy as np

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
    X = np.linalg.inv((np.eye(N)-alpha*A-beta*B)) @ curve
    return X

def euclidianLength(p1,p2):
    length = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
    return length

hand = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/hand_noisy.txt")


alpha1 = 100
beta1 = 0
alpha2 = 0
beta2 = 100 

X1 = curveSmoothImplicit(hand,alpha1,beta1)
X2 = curveSmoothImplicit(hand,alpha2,beta2)

l1 = euclidianLength(X1[0,:],X1[X1.shape[0]-1,:])
l2 = euclidianLength(X2[0,:],X2[X2.shape[0]-1,:])

for i in range(1,hand.shape[0]):
    l1 += euclidianLength(X1[i-1,:],X1[i,:])
    l2 += euclidianLength(X2[i-1,:],X2[i,:])

d = l1 - l2

print(d)
