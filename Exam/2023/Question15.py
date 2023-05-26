import numpy as np
import matplotlib.pyplot as plt

W1 = np.array([[0.7,9,-9.1],
               [0.3,-10.1,10]])
W2 = np.array([[0.2,5.6,-5.9],
               [-0.5,-4.6,5.4]])


n = 5000
points = 2*np.random.random([2,n])-1


h_l1 = W1 @ np.vstack([np.ones([1,n]),points])
h_l1[h_l1<0] = 0



y_hat = W2 @ np.vstack([np.ones([1,n]),h_l1])

y = (np.exp(y_hat)/np.sum(np.exp(y_hat),axis=0))

classes = np.argmax(y,axis=0)

c0 = points[:,classes==0] 
c1 = points[:,classes==1] 



print(c1.shape)


plt.figure()
plt.scatter(c0[0,:],c0[1,:],label="class1")
plt.scatter(c1[0,:],c1[1,:],label="class2")
plt.legend()



plt.show()

