import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy import ndimage 


Im1_c = io.imread("optical_flow_data/Army/frame10.png")
Im2_c = io.imread("optical_flow_data/Army/frame11.png")
Im1 = color.rgb2gray(Im1_c)
Im2 = color.rgb2gray(Im2_c)


diffKernel = np.array([[-1,0,1],
                       [-1,0,1],
                       [-1,0,1]])




Im1x = ndimage.convolve(Im1,-diffKernel)
Im1y = ndimage.convolve(Im1,-diffKernel.T)
Im1xx = ndimage.convolve(Im1x,diffKernel)
Im1yy = ndimage.convolve(Im1y,-diffKernel.T)


Im2x = ndimage.convolve(Im2,diffKernel)
Im2y = ndimage.convolve(Im2,diffKernel.T)

Imt = Im2-Im1


## Good 

alpha = 0.7
iterations = 100
n=5
meanKernel = np.ones([n,n])/(n*n)
u1 = np.zeros(Im1.shape)
v1 = np.zeros(Im1.shape)

for i in range(iterations):
    u_bar = ndimage.convolve(u1,meanKernel)
    v_bar = ndimage.convolve(v1,meanKernel)
    u1 = u_bar - ((Im1x*(Im1x*u_bar+Im1y*v_bar+Imt))/(alpha**2 + Im1x**2 + Im1y**2))
    v1 = v_bar - ((Im1y*(Im1x*u_bar+Im1y*v_bar+Imt))/(alpha**2 + Im1x**2 + Im1y**2))

# Bad alpha
alpha = 0.05
iterations = 100
n=5
meanKernel = np.ones([n,n])/(n*n)
u2 = np.zeros(Im1.shape)
v2 = np.zeros(Im1.shape)

for i in range(iterations):
    u_bar = ndimage.convolve(u2,meanKernel)
    v_bar = ndimage.convolve(v2,meanKernel)
    u2 = u_bar - ((Im1x*(Im1x*u_bar+Im1y*v_bar+Imt))/(alpha**2 + Im1x**2 + Im1y**2))
    v2 = v_bar - ((Im1y*(Im1x*u_bar+Im1y*v_bar+Imt))/(alpha**2 + Im1x**2 + Im1y**2))

#Few iterations
alpha = 0.7
iterations = 5
n=5
meanKernel = np.ones([n,n])/(n*n)
u3 = np.zeros(Im1.shape)
v3 = np.zeros(Im1.shape)

for i in range(iterations):
    u_bar = ndimage.convolve(u3,meanKernel)
    v_bar = ndimage.convolve(v3,meanKernel)
    u3 = u_bar - ((Im1x*(Im1x*u_bar+Im1y*v_bar+Imt))/(alpha**2 + Im1x**2 + Im1y**2))
    v3 = v_bar - ((Im1y*(Im1x*u_bar+Im1y*v_bar+Imt))/(alpha**2 + Im1x**2 + Im1y**2))



#Big window
alpha = 0.7
iterations = 100
n=50
meanKernel = np.ones([n,n])/(n*n)
u4 = np.zeros(Im1.shape)
v4 = np.zeros(Im1.shape)

for i in range(iterations):
    u_bar = ndimage.convolve(u4,meanKernel)
    v_bar = ndimage.convolve(v4,meanKernel)
    u4 = u_bar - ((Im1x*(Im1x*u_bar+Im1y*v_bar+Imt))/(alpha**2 + Im1x**2 + Im1y**2))
    v4 = v_bar - ((Im1y*(Im1x*u_bar+Im1y*v_bar+Imt))/(alpha**2 + Im1x**2 + Im1y**2))






x,y = np.meshgrid(np.arange(0,Im1.shape[1],25),np.arange(0,Im1.shape[0],25))

plt.figure()

plt.subplot(221)
plt.imshow(Im1_c)
plt.quiver(x,y,u1[y,x],-v1[y,x],color='red')
plt.title("Nice  values")

plt.subplot(222)
plt.imshow(Im1_c)
plt.quiver(x,y,u2[y,x],-v2[y,x],color='red')
plt.title("Low alpha")

plt.subplot(223)
plt.imshow(Im1_c)
plt.quiver(x,y,u3[y,x],-v3[y,x],color='red')
plt.title("Few iterations")

plt.subplot(224)
plt.imshow(Im1_c)
plt.quiver(x,y,u4[y,x],-v4[y,x],color='red')
plt.title("Large window")







plt.show()




