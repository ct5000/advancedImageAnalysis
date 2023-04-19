import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy import ndimage 


Im1 = io.imread("optical_flow_data/Basketball/frame10.png")
Im2 = io.imread("optical_flow_data/Basketball/frame11.png")
Im1 = color.rgb2gray(Im1)
Im2 = color.rgb2gray(Im2)


diffKernel = np.array([[-1,0,1]])
meanKernel = np.ones([3,3])/9



Im1x = ndimage.convolve(Im1,-diffKernel)
Im1y = ndimage.convolve(Im1,-diffKernel.T)

Im2x = ndimage.convolve(Im2,diffKernel)
Im2y = ndimage.convolve(Im2,diffKernel.T)

Imt = Im2-Im1

alpha = 0.3
iterations = 40

u = np.zeros(Im1.shape)
v = np.zeros(Im1.shape)


for i in range(iterations):
    u_bar = ndimage.convolve(u,meanKernel)
    v_bar = ndimage.convolve(v,meanKernel)
    u = u_bar - ((Im1x*(Im1x*u_bar+Im1y*v_bar+Imt))/(alpha**2 + Im1x**2 + Im2y**2))
    v = v_bar - ((Im1y*(Im1x*u_bar+Im1y*v_bar+Imt))/(alpha**2 + Im1x**2 + Im2y**2))


plt.figure()
plt.subplot(211)
plt.imshow(Im1)

plt.subplot(212)
plt.imshow(Im2)





plt.figure()
plt.imshow(Im1)


x,y = np.meshgrid(np.arange(0,Im1.shape[1],50),np.arange(0,Im1.shape[0],50))

plt.quiver(x,y,u[y,x],-v[y,x],color='red')













plt.show()




