import numpy as np
import skimage.io
import scipy

def window_std(I):
    window = np.ones(shape=(5, 5))/25
    K_I2 = scipy.ndimage.convolve(I**2, window, mode='reflect')
    KI_2 = scipy.ndimage.convolve(I, window, mode='reflect')**2
    return np.sqrt(K_I2 - KI_2)

I = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/blended.png")/255

sigma_text = 0.05
sigma_smooth = 0.01
sigma = np.array([sigma_text,sigma_smooth],dtype=float)
S = window_std(I)


U = (S.reshape(S.shape+(1,)) - sigma.reshape(1,1,-1))**2
S0 = np.argmin(U, axis=2)


#V1 = np.sum(np.sum(np.power(sigma_text-S,2) + np.power(sigma_smooth-S,2)))
V1 = ((sigma[S0]-S)**2).sum()


print(V1)














