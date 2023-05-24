import numpy as np






A = np.diag(-2*np.ones([5])) + np.diag(np.ones([4]),1) + np.diag(np.ones([4]),-1)
B = np.diag(-6*np.ones([5])) + np.diag(4*np.ones([4]),1) + np.diag(4*np.ones([4]),-1) + np.diag(-np.ones([3]),2) + np.diag(-np.ones([3]),-2)
print(A)
print(B)

alpha = 0.05
beta = 0.1
invert = np.linalg.inv(np.identity(5)-alpha*A-beta*B)

points = np.array([[3.5,0.2],[1.4,1.1],[0.1,2.9],[1.2,5.4],[3.3,7.1]])

print(invert@points)



