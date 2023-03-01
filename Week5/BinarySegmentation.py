import skimage.io
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import maxflow


def binarySegmentImage(Im,mu,beta):
    w_s = (Im-mu[0])**2 # source weight
    w_t = (Im-mu[1])**2 # sink weights
    rows,cols = Im.shape
    N = rows*cols

    # Create a graph with integer capacities.
    g = maxflow.Graph[float]()
    # Add (non-terminal) nodes and retrieve an index for each node
    nodes = g.add_nodes(N)
    # Create edges between nodes
    for i in range(rows-1):
        for j in range(cols-1):
            g.add_edge(nodes[i*cols+j], nodes[i*cols+j+1], beta, beta)
            g.add_edge(nodes[i*cols+j], nodes[i*cols+j+cols], beta, beta)
    # Set the capacities of the terminal edges.
    for i in range(rows):
        for j in range(cols):
            g.add_tedge(nodes[i*cols+j], (Im[i,j]-mu[1])**2, (Im[i,j]-mu[0])**2)
    # Run the max flow algorithm
    flow = g.maxflow()
    print(flow)
    Im_seg = np.zeros([rows,cols])
    for i in range(rows):
        for j in range(cols):
            Im_seg[i,j] = g.get_segment(nodes[i*cols+j])
    Im_seg[Im_seg==1] = 255
    return Im_seg




Im = skimage.io.imread('DTU_noisy.png').astype(float) /255
Im_true = skimage.io.imread('DTU.png')


mu = [90/255,170/255]
beta = 0.1



Im_seg = binarySegmentImage(Im,mu,beta)

 
plt.figure()
plt.imshow(Im_seg)

plt.figure()
plt.imshow(Im_true)

err = sum(sum(abs(Im_true - Im_seg))) / 255
print(err)




Im_bone = skimage.io.imread('V12_10X_x502.png').astype(float) / (2**16 - 1)

mu_bone = [0.4,0.7]
beta_bone = 0.05

Im_seg_bone = binarySegmentImage(Im_bone,mu_bone,beta_bone)

plt.figure()
plt.imshow(Im_seg_bone)

plt.figure()
plt.imshow(Im_bone)


plt.show()

