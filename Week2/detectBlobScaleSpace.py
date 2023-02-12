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



def detectBlobs(Im,numScales,t,breadth,threshold):
    scaleSpace = np.zeros([numScales,Im.shape[0],Im.shape[1]])
    scaleSpace[0,:,:] = Im
    Laplacian3D = np.zeros([numScales,Im.shape[0],Im.shape[1]])
    g, _, ddg, _ = guassianKernel(t,breadth)
    for i in range(1,numScales):
        
        scaleSpace[i,:,:] = ndimage.convolve(ndimage.convolve(scaleSpace[i-1,:,:],g),g.T)
        Im_lapx = ndimage.convolve(ndimage.convolve(scaleSpace[i-1,:,:],ddg),g.T)
        Im_lapy = ndimage.convolve(ndimage.convolve(scaleSpace[i-1,:,:],ddg.T),g)
        Laplacian3D[i-1,:,:] = (Im_lapx + Im_lapy)*t*i
        '''
        g, _, ddg, _ = guassianKernel(t*i,breadth)
        Im_lapx = ndimage.convolve(ndimage.convolve(Im,ddg),g.T)
        Im_lapy = ndimage.convolve(ndimage.convolve(Im,ddg.T),g)
        Laplacian3D[i-1,:,:] = (Im_lapx + Im_lapy)*t*i
        '''
        #print((sigma**2)*i)
        #plt.figure(i)
        #plt.imshow(Laplacian3D[i-1,:,:])
    #plt.show()
    local_max = peak_local_max(Laplacian3D,min_distance = 1,threshold_abs=threshold)
    local_min = peak_local_max(-Laplacian3D,min_distance = 1,threshold_abs=threshold)
    return np.r_[local_max,local_min]

'''
Im_test = np.float32(skimage.io.imread("test_blob_varying.png"))

blobs = detectBlobs(Im_test,1000,1,5,75)

print(blobs.shape)
print(blobs)

plt.figure(5)
plt.imshow(Im_test)

ax = plt.gca()
for i in range(blobs.shape[0]):
    #print(blobs[i,:])
    circle = plt.Circle((blobs[i,2],blobs[i,1]),1*np.sqrt(2*(blobs[i,0]+1)),color='red',fill=False)
    ax.add_artist(circle)
'''


Im_CT = np.float32(skimage.io.imread("CT_lab_med_res.png"))

blobs = detectBlobs(Im_CT,200,1,5,4000)

print(blobs.shape)
#print(blobs)
print(np.max(blobs[:,0]))
plt.figure(6)
plt.imshow(Im_CT)
r_sum = 0
ax = plt.gca()
for i in range(blobs.shape[0]):
    #print(blobs[i,:])
    circle = plt.Circle((blobs[i,2],blobs[i,1]),1*np.sqrt(2*(blobs[i,0]+1)),color='red',fill=False)
    r_sum += 1*np.sqrt(2*(blobs[i,0]+1))
    ax.add_artist(circle)

r_mean = r_sum/blobs.shape[0]
print(r_mean)
print(blobs.shape[0])
plt.show()






