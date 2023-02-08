import numpy as np
from scipy import linalg, interpolate, ndimage
import skimage.io
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max

def guassianKernel(sigma, breadth):
    s = breadth*sigma
    x = np.arange(-s,s+1)
    x = np.reshape(x,[1,len(x)])
    g = 1 / (sigma*np.sqrt(2*np.pi))*np.exp((-x**2/(2*sigma**2)))
    dg = -x / sigma**2 * g
    ddg = dg * (-x / sigma**2) + g * (-1 / sigma**2)
    return g, dg, ddg, x




g, dg, ddg, x = guassianKernel(10,5)
'''
plt.figure(1)
plt.plot(x[0,:],ddg[0,:])
'''
Im_test = np.float32(skimage.io.imread("test_blob_uniform.png"))


Im_conv = ndimage.convolve(ndimage.convolve(Im_test,ddg),ddg.T)

'''
plt.figure(2)
plt.imshow(Im_test)
plt.figure(3)
plt.imshow(Im_conv)
'''

Im_lapx = ndimage.convolve(ndimage.convolve(Im_test,ddg),g.T)
Im_lapy = ndimage.convolve(ndimage.convolve(Im_test,ddg.T),g)
Im_lap = Im_lapx + Im_lapy

'''
plt.figure(4)
plt.imshow(Im_lap)
'''


def detectBlobs(Im, sigma, breadth,threshold):
    g, dg, ddg, x = guassianKernel(sigma,breadth)
    Im_lapx = ndimage.convolve(ndimage.convolve(Im,ddg),g.T)
    Im_lapy = ndimage.convolve(ndimage.convolve(Im,ddg.T),g)
    Im_lap = (Im_lapx + Im_lapy)*sigma**2
    local_max = peak_local_max(Im_lap,min_distance = 1,threshold_abs=threshold)
    local_min = peak_local_max(-Im_lap,min_distance = 1,threshold_abs=threshold)
    return Im_lap, local_max, local_min


Im22, local_max, local_min = detectBlobs(Im_test,19,5,50)

blobs = np.r_[local_max,local_min]
plt.figure(5)
plt.imshow(Im_test)
ax = plt.gca()
print(blobs.shape)
for i in range(blobs.shape[0]):
    print(blobs[i,:])
    circle = plt.Circle((blobs[i,1],blobs[i,0]),np.sqrt(2)*20,color='red',fill=False)
    ax.add_artist(circle)
    







plt.show()


