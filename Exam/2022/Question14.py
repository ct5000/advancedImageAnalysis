import numpy as np


data = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/in_t_out.txt")

P = data[:,0:3]
T = data[:,3].astype(int)
Y_hat = data[:,4:]

Y = (np.exp(Y_hat.T)/np.sum(np.exp(Y_hat.T),axis=0,keepdims=True)).T
print(Y)

loss = 0
for i in range(Y.shape[0]):
    loss -= np.log(Y[i,T[i]])
    '''
    if T[i]== 0 and Y[i,0] > Y[i,1]:
        loss -= np.log(Y[i,0]) 
        print(i)
    if T[i]== 1 and Y[i,0] < Y[i,1]:
        loss -= np.log(Y[i,1]) 
        print(i)
    '''



print(loss)
loss2 = -np.log(Y[range(len(Y)), T]).sum()
print(loss2)
