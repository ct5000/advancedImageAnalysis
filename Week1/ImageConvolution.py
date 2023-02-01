import numpy as np
from scipy import linalg, interpolate, ndimage
import skimage.io
import matplotlib.pyplot as plt
import cv2



def guassianKernel(sigma, breadth):
    s = breadth*sigma
    x = np.arange(-s,s+1)
    x = np.reshape(x,[1,len(x)])
    g = 1 / (sigma*np.sqrt(2*np.pi))*np.exp((-x**2/(2*sigma**2)))
    dg = -x / sigma**2 * g
    return g, dg, x



g, dg, x = guassianKernel(4.5,5)
g2D = g*g.T
#Showing kernel


'''
plt.figure(1)
plt.plot(x[0,:],g[0,:])

plt.figure(2)
plt.plot(x[0,:],dg[0,:])
'''


Im = np.float32(skimage.io.imread("fibres_xcth.png"))
#skimage.io.imshow(Im)
#plt.show()


con2D = ndimage.convolve(Im,g2D)
con1D = ndimage.convolve(ndimage.convolve(Im,g),g.T)

# Plot the two kinds of convolution
'''
plt.figure(3)
plt.subplot(121)
plt.imshow(con2D)
plt.subplot(122)
plt.imshow(con1D)


conDiff = con2D - con1D
plt.figure(4)
plt.imshow(conDiff)

'''

derivImCon = ndimage.convolve(ndimage.convolve(Im,dg.T),g)
kerDif = np.array([[0.5,0,-0.5]])
print(kerDif.shape)
print(dg.shape)
derivImNorm = ndimage.convolve(ndimage.convolve(Im,kerDif.T),g)

#Show diff image
'''
plt.figure(5)
plt.subplot(121)
plt.imshow(derivImCon)
plt.subplot(122)
plt.imshow(derivImNorm)

derivDiff = derivImCon - derivImNorm
plt.figure(6)
plt.imshow(derivDiff)

'''

g2, dg2, x2 = guassianKernel(np.sqrt(20),5)
g3, dg3, x3 = guassianKernel(np.sqrt(2),5)

convT20 = ndimage.convolve(ndimage.convolve(Im,g2),g2.T)

convT2 = Im
for n in range(10):
    convT2 = ndimage.convolve(ndimage.convolve(convT2,g3),g3.T)

convDiff2 = convT20 - convT2
plt.figure(7)
plt.imshow(convDiff2)


g4, dg4, x4 = guassianKernel(np.sqrt(10),5)
convT10 = Im
for n in range(2):
    convT10 = ndimage.convolve(ndimage.convolve(convT10,g4),g4.T)
convDiff3 = convT20 - convT10
plt.figure(8)
plt.imshow(convDiff3)




plt.show()










