import numpy as np
import matplotlib.pyplot as plt

W1 = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/W1.txt")
W2 = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/W2.txt")
W3 = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/W3.txt")


n = 5000
points = 2*np.random.random([2,n])-1


h_l1 = W1.T @ np.vstack([np.ones([1,n]),points])
h_l1[h_l1<0] = 0

h_l2 = W2.T @ np.vstack([np.ones([1,n]),h_l1])
h_l2[h_l2<0] = 0


y_hat = W3.T @ np.vstack([np.ones([1,n]),h_l2])

y = (np.exp(y_hat)/np.sum(np.exp(y_hat),axis=0))

classes = np.argmax(y,axis=0)

c0 = points[:,classes==0] 
c1 = points[:,classes==1] 
c2 = points[:,classes==2] 
c3 = points[:,classes==3] 


print(c1.shape)


plt.figure()
plt.plot(c0[0,:],c0[1,:])
plt.plot(c1[0,:],c1[1,:])
plt.plot(c2[0,:],c2[1,:])
plt.plot(c3[0,:],c3[1,:])


plt.show()

