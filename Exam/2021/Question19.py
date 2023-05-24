import numpy as np


W1 = np.array([[0.2,-1.3],
               [-0.3,1.8],
               [-1.7,1.6]])
W2 = np.array([[-1.4,1.5,-0.5,0.9],
               [0.2,1.2,-0.9,1.7]])

x = np.array([[1],[2.5]])

h_u = W1@x
h = np.vstack([1,(h_u>0)*h_u])
print(h)

y_hat = W2@h
y = np.exp(y_hat)/(np.sum(np.exp(y_hat)))

print(y_hat)
print(y)





