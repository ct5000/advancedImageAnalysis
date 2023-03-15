import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import slgbuilder
import scipy.interpolate

np.bool = bool

Im_comb = skimage.io.imread('dental_slices.tif').astype(np.int32)
print(Im_comb.shape)

Ims = [np.reshape(Im_comb[i,:,:],[Im_comb.shape[1],Im_comb.shape[2]]) for i in range(Im_comb.shape[0])]

'''
fig, ax = plt.subplots(2,3)


for i in range(len(Ims)):
    ax[int(i/3),i%3].imshow(Ims[i],cmap='gray')

'''
fig, ax = plt.subplots(1,2)
ax[0].imshow(Ims[0], cmap='gray')

a = 180 # number of angles for unfolding
angles = np.arange(a)*2*np.pi/a # angular coordinate

center = (np.array(Ims[0].shape)-1)/2
r = int(min(Ims[0].shape)/2)
radii = np.arange(r) + 1 #radial coordinate for unwrapping

X = center[0] + np.outer(radii,np.cos(angles))
Y = center[1] + np.outer(radii,np.sin(angles))

F = scipy.interpolate.interp2d(np.arange(Ims[0].shape[0]), np.arange(Ims[0].shape[1]), Ims[0])
#F = scipy.interpolate.RegularGridInterpolator((np.arange(Ims[0].shape[0]), np.arange(Ims[0].shape[1])), Ims[0])

U = np.array([F(p[0],p[1]) for p in np.c_[Y.ravel(),X.ravel()]])
U = U.reshape((r,a)).astype(np.int32)


layers = [slgbuilder.GraphObject(U),slgbuilder.GraphObject(U)] # no on-surface cost
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)

# Addin regional costs, 
# the region in the middle is bright compared to two darker regions.
helper.add_layered_region_cost(layers[0], U, 255-U)
helper.add_layered_region_cost(layers[1], 255-U, U)

# Adding geometric constrains
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=10, wrap=False)  
helper.add_layered_containment(layers[0], layers[1], min_margin=1)

# Cut
helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]


# Visualization
ax[1].imshow(U, cmap='gray')
for line in segmentation_lines:
    ax[1].plot(line, 'r')




plt.show()