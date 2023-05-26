import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy import ndimage


def guassianKernel(sigma, breadth):
    s = breadth*sigma
    x = np.arange(-s,s+1)
    x = np.reshape(x,[1,len(x)])
    g = 1 / (sigma*np.sqrt(2*np.pi))*np.exp((-x**2/(2*sigma**2)))
    dg = -x / sigma**2 * g
    ddg = dg * (-x / sigma**2) + g * (-1 / sigma**2)
    return g, dg, ddg, x



img_org = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/wood.png").astype(np.float32)
img_lap = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/wood_LoG.tif")
plt.figure()
plt.imshow(img_lap)
print(img_org.shape)


sigma = [1.2,2,2.4,3.3,4.8,5.5,6.4,8.3,13.4,22.8,30.3]

for i in range(len(sigma)):
    g, dg, ddg, x = guassianKernel(sigma[i],4)
    Im_lapx = ndimage.convolve(ndimage.convolve(img_org,ddg),g.T)
    Im_lapy = ndimage.convolve(ndimage.convolve(img_org,ddg.T),g)
    Im_lap = Im_lapx + Im_lapy
    diff = abs(img_lap-Im_lap)
    print(i, sigma[i])
    print(np.sum(np.sum(diff)))
    if (i==5):
        plt.figure()
        plt.imshow(Im_lap)
        plt.figure()
        plt.imshow(diff)

plt.show()




