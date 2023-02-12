import numpy as np
from scipy import linalg, interpolate, ndimage
import skimage.io
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max

def guassianKernel(t, breadth):
    sigma = np.sqrt(t)
    s = breadth*sigma
    x = np.arange(-s,s+1)
    x = np.reshape(x,[1,len(x)])
    g = 1 / (sigma*np.sqrt(2*np.pi))*np.exp((-x**2/(2*sigma**2)))
    dg = -x / sigma**2 * g
    ddg = dg * (-x / sigma**2) + g * (-1 / sigma**2)
    return g, dg, ddg, x



Im_CT = np.float32(skimage.io.imread("CT_lab_med_res.png"))
g, _, ddg, _ = guassianKernel(3,5)

Im_smooth = ndimage.convolve(ndimage.convolve(Im_CT,g),g.T)

local_max = peak_local_max(Im_smooth,min_distance = 1,threshold_abs=30000)
print(local_max.shape)
plt.figure(1)
plt.imshow(Im_smooth)
plt.scatter(local_max[:,1],local_max[:,0],s=2,color='red')



def LaplacianStateSpace(Im,numScales,t,breadth):
    scaleSpace = np.zeros([numScales,Im.shape[0],Im.shape[1]])
    scaleSpace[0,:,:] = Im
    Laplacian3D = np.zeros([numScales,Im.shape[0],Im.shape[1]])
    g, _, ddg, _ = guassianKernel(t,breadth)
    for i in range(1,numScales):
        
        scaleSpace[i,:,:] = ndimage.convolve(ndimage.convolve(scaleSpace[i-1,:,:],g),g.T)
        Im_lapx = ndimage.convolve(ndimage.convolve(scaleSpace[i-1,:,:],ddg),g.T)
        Im_lapy = ndimage.convolve(ndimage.convolve(scaleSpace[i-1,:,:],ddg.T),g)
        Laplacian3D[i-1,:,:] = (Im_lapx + Im_lapy)*t*i

    return Laplacian3D


LaplaceStateSpace = LaplacianStateSpace(Im_CT,10,1,5)
scales = np.zeros([local_max.shape[0],1])
for i in range(local_max.shape[0]):
    scales[i] = np.argmin(LaplaceStateSpace[:,local_max[i,0],local_max[i,1]])


plt.figure(2)
plt.imshow(Im_smooth)
ax = plt.gca()
for i in range(local_max.shape[0]):
    circle = plt.Circle((local_max[i,1],local_max[i,0]),1*np.sqrt(2*(scales[i]+1)),color='red',fill=False)
    ax.add_artist(circle)







plt.show()








