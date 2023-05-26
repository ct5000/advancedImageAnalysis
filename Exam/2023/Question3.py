import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy import ndimage
from skimage.feature import peak_local_max
import skimage.feature


def guassianKernel(sigma, breadth):
    s = breadth*sigma
    x = np.arange(-s,s+1)
    x = np.reshape(x,[1,len(x)])
    g = 1 / (sigma*np.sqrt(2*np.pi))*np.exp((-x**2/(2*sigma**2)))
    dg = -x / sigma**2 * g
    ddg = dg * (-x / sigma**2) + g * (-1 / sigma**2)
    return g, dg, ddg, x

def detectBlobs(Im, sigma, breadth,threshold):
    g, dg, ddg, x = guassianKernel(sigma,breadth)
    Im_lapx = ndimage.convolve(ndimage.convolve(Im,ddg),g.T)
    Im_lapy = ndimage.convolve(ndimage.convolve(Im,ddg.T),g)
    Im_lap = (Im_lapx + Im_lapy)*sigma**2
    local_max = peak_local_max(Im_lap,min_distance = 1,threshold_abs=threshold)
    local_min = peak_local_max(-Im_lap,min_distance = 1,threshold_abs=threshold)
    return Im_lap, local_max, local_min


img_org = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2023/DATA_2023/fiber.png").astype(np.float32)


Im_lap, local_max, local_min = detectBlobs(img_org, 3,4,0)
print("My own")
print(local_max.shape)
print(local_min.shape)

print("Skimage")

blobs_light = skimage.feature.blob_log(img_org,min_sigma=3,max_sigma=3)
blobs_dark = skimage.feature.blob_log(-img_org,min_sigma=3,max_sigma=3)

print(blobs_light.shape)
print(blobs_dark.shape)

