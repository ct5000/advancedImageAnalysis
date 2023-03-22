import numpy as np
import make_data


x0 = 1
x1 = -120
w11_1 = -0.1
w10_1 = -10
w21_1 = 0.15

w10_2 = 4
w11_2 = 0.5
w21_2 = 2.1
w22_2 = 0.2

h0 = 1
h1 = np.max([0,x0*w10_1+x1*w11_1])
h2 = np.max([0,x1*w21_1])

y1 = w10_2 * h0 + w11_2 * h1
y2 = w21_2 * h2 + w21_2 * h1

y1_norm = np.exp(y1)/(np.exp(y1)+np.exp(y2))
y2_norm = np.exp(y2)/(np.exp(y1)+np.exp(y2))
print(y1_norm)

loss = -np.log(y2_norm)
print(loss)


delta2_2 = y1
dLoss_dw11_2 = h1*delta2_2

print(dLoss_dw11_2)

