import numpy as np
from scipy import linalg, interpolate, ndimage
import skimage.io
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
import math
import local_features
from sklearn.cluster import KMeans

def prepareLabelIm(labels):
    uniqueLabels = np.unique(labels)
    labelsIm = np.zeros([labels.shape[0],labels.shape[1],len(uniqueLabels)])
    for i in range(len(uniqueLabels)):
        labelsIm[:,:,i] = labels == uniqueLabels[i]
    return np.reshape(labelsIm,[labels.shape[0]*labels.shape[1],len(uniqueLabels)])


def guassianKernel(sigma, breadth):
    s = breadth*sigma
    x = np.arange(-s,s+1)
    x = np.reshape(x,[1,len(x)])
    g = 1 / (sigma*np.sqrt(2*np.pi))*np.exp((-x**2/(2*sigma**2)))
    dg = -x / sigma**2 * g
    return g, dg, x


Im = skimage.io.imread("3labels/training_image.png")

plt.figure()
plt.imshow(Im,cmap='gray')

features = local_features.get_gauss_feat_multi(Im,sigma=[1])
features = np.reshape(features,[features.shape[0],features.shape[1]*features.shape[2]])

print(features.shape)

labels = skimage.io.imread("3labels/training_labels.png")


labelsIm = prepareLabelIm(labels)
print(labelsIm.shape)

'''
plt.figure()
plt.subplot(1,2,1)
plt.imshow(labelsIm[0,:,:])
plt.subplot(1,2,2)
plt.imshow(labelsIm[1,:,:])
'''
nClusters= 400
idx = np.random.choice(labelsIm.shape[0],size=int(np.round(features.shape[0]*0.1)),replace=False)

randomFeatures = features[idx,:]
randomLabels = labelsIm[idx,:]

kmeans = KMeans(n_clusters = nClusters,n_init='auto').fit(randomFeatures)

print(kmeans.labels_.shape)

clusterProb = np.zeros([labelsIm.shape[1],nClusters])

c, index, counts = np.unique(kmeans.labels_,return_inverse = True, return_counts=True)


i = 0
indexCur = 0
while i < nClusters:
    labelOccurence = np.zeros([labelsIm.shape[1]])
    for k in range(counts[i]):
        labelOccurence += randomLabels[index[indexCur],:]
        indexCur += 1
    clusterProb[:,i] = labelOccurence / counts[i]
    i += 1


testIm = skimage.io.imread("3labels/testing_image.png")
r, c = testIm.shape
testFeatures = local_features.get_gauss_feat_multi(testIm,sigma=[1])
testFeatures = np.reshape(testFeatures,[testFeatures.shape[0],testFeatures.shape[1]*testFeatures.shape[2]])
testClusters = kmeans.predict(testFeatures)
testClusters = np.reshape(testClusters,[r,c])
print(testClusters.shape)

#probIm = np.zeros([r,c,labelsIm.shape[1]])
probIm = clusterProb[:,testClusters]

probImSmooth = np.zeros(probIm.shape)
kernelSize = 5
kernel = np.ones([kernelSize,kernelSize])*(1/(kernelSize**2))


for i in range(probIm.shape[0]):
    probImSmooth[i,:,:] = ndimage.convolve(probIm[i,:,:],kernel)

probImSmooth = probImSmooth/(np.sum(probImSmooth,axis=0))

plt.figure()
plt.subplot(2,2,1)
plt.imshow(probIm[0,:,:])
plt.subplot(2,2,2)
plt.imshow(probImSmooth[0,:,:])
plt.subplot(2,2,3)
plt.imshow(probIm[1,:,:])
plt.subplot(2,2,4)
plt.imshow(probImSmooth[1,:,:])


finalIm = np.argmax(probImSmooth,axis=0)*255

plt.figure()
plt.imshow(finalIm)


plt.show()


