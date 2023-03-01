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


def segmentationBoundary(segmentation):
    l1 = (segmentation[:-1,:] != segmentation[1:,:]).sum()
    l2 = (segmentation[:,:-1] != segmentation[:,1:]).sum()
    return l1 + l2

def segmentation_energy(S, D, mu, beta):
    # TODO -- add your code here
    for i in range(len(mu)):
        S[S==i] = mu[i]
    # likelihood energy
    U1 = sum(sum((S-D)**2))
    
    # prior energy

    U2 = beta * segmentationBoundary(S)
    
    return U1, U2


mu = [2, 5, 10]
beta = 10

D = np.array([[1,2,6,4,10,8],
              [4,1,3,5,9,7],
              [5,2,3,5,4,8]])

# Q1

segment_23 = binarySegmentImage(D,[2,5],beta)
print(segment_23)
segment_23 = binarySegmentImage(D,[5,10],beta)
print(segment_23)

S1 = np.array([[2,2,5,5,10,10],
               [2,2,5,5,5,5],
               [2,2,5,5,5,10]])

V1,V2 = segmentation_energy(S1,D,mu,beta)
print("Q1 and Q3")
print(f'Likelihood: {V1}')
print(f'Prior: {V2}')
print(f'Posterior {V1+V2}')


# Q2
S2 = np.array([[2,2,5,5,10,10],
              [2,2,5,5,10,10],
              [2,2,5,5,10,10]])

V1,V2 = segmentation_energy(S2,D,mu,beta)
print("Q2")
print(V1)











