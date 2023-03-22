import numpy as np
import pandas
import make_data
import matplotlib.pyplot as plt



def neuralNetwork(X, T, n_inputs,n_outputs,n_hidden_notes,n_hidden_layers,learning_rate,train_rounds):
    W = [np.random.normal(size=[n_inputs+1,n_hidden_notes])]
    for i in range(n_hidden_layers-1):
        W.append(np.random.normal(size=[n_hidden_notes+1,n_hidden_notes]))
    W.append(np.random.normal(size=[n_hidden_notes+1,n_outputs]))


    for k in range(train_rounds):
        test_point = np.reshape(X[:,k],[2,1])
        # Forward part
        
        h = []
        point_before = test_point
        for i in range(n_hidden_layers):
            z =  W[i].T @ np.vstack([point_before, 1])
            h.append(np.maximum(z,0))
            point_before = h[i]
        
        y_hat = W[n_hidden_layers].T @ np.vstack([point_before, 1])
        y = np.exp(y_hat)
        y = y/(y.sum())
        
        # Backward part
        Q = [None] * (n_hidden_layers+1) 
        delta = [None] * (n_hidden_layers+1) 
        
        delta[n_hidden_layers] = y-np.reshape(T[:,k],[2,1])
        Q[n_hidden_layers] = np.vstack([np.reshape(h[n_hidden_layers-1],[n_hidden_notes,1]),1]) @ delta[n_hidden_layers].T 
        

        for i in range(n_hidden_layers-1,0,-1):
            a_mark = h[i] > 0
            delta[i] = a_mark * (W[i+1][0:n_hidden_notes,:]@delta[i+1])
            Q[i] = np.vstack([h[i-1], 1]) @ delta[i].T
        a_mark = h[0] > 0
        delta[0] = a_mark * (W[1][0:n_hidden_notes,:]@delta[1])
        Q[0] = np.vstack([test_point, 1]) @ delta[0].T
        
        for i in range(n_hidden_layers+1):
            W[i] = W[i] - learning_rate*Q[i]

        return W


n_points = 200
X, T, x, dim = make_data.make_data(3,n=n_points)
mean = np.mean(X,axis=1)
variance = np.var(X,axis=1)

class1 = X[:,T[0,:]]
class2 = X[:,T[1,:]]
plt.figure()
plt.plot(class1[0,:],class1[1,:],'b.')
plt.plot(class2[0,:],class2[1,:],'r.')


mean = np.reshape(np.mean(X,axis=1),[2,1])
std = np.reshape(np.std(X,axis=1),[2,1])

X = (X - mean) / std

class1 = X[:,T[0,:]]
class2 = X[:,T[1,:]]
plt.figure()
plt.plot(class1[0,:],class1[1,:],'b.')
plt.plot(class2[0,:],class2[1,:],'r.')

n_inputs = 2
n_hidden_layers = 1
n_hidden_notes =  3
n_outputs = 2
learning_rate = 0.01

train_rounds = int(n_points*2*0.8)

'''
W = [0.1*np.ones([n_inputs+1,n_hidden_notes])]
for i in range(n_hidden_layers-1):
    W.append(0.1*np.ones([n_hidden_notes+1,n_hidden_notes]))
W.append(0.1*np.ones([n_hidden_notes+1,n_outputs]))
'''


W = neuralNetwork(X, T, n_inputs,n_outputs,n_hidden_notes,n_hidden_layers, learning_rate,train_rounds)

print(W)
    
test_rounds = n_points*2 - train_rounds
n_succes = 0
for k in range(test_rounds):
    test_point = np.reshape(X[:,train_rounds+k],[2,1])
    # Forward part
    
    h = []
    point_before = test_point
    for i in range(n_hidden_layers):
        z =  W[i].T @ np.vstack([point_before, 1])
        h.append(np.maximum(z,0))
        point_before = h[i]
    y_hat = W[len(h)].T @ np.vstack([point_before, 1])
    y = np.exp(y_hat)
    y = y/(y.sum())

    if y[0] > y[1]:
        if T[0,train_rounds+k]:
            n_succes += 1
    elif y[0] < y[1]:
        if T[1,train_rounds+k]:
            n_succes += 1

print(n_succes/test_rounds)


plt.show()