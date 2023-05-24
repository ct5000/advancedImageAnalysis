import numpy as np




y_hat = np.array([0.5,8.2,6.9,-0.1,0.3])

y = np.exp(y_hat)/(np.sum(np.exp(y_hat)))

print(y)

print(-np.log(y))











