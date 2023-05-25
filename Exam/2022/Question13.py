import numpy as np
import slgbuilder
import matplotlib.pyplot as plt
import skimage.io 

np.bool = bool
I = np.loadtxt("/home/christian/advancedImageAnalysis/Exam/2022/EXAM_MATERIAL_2022/cost.txt")


fig, ax = plt.subplots(1,4)
ax[0].imshow(I, cmap='gray')
ax[0].set_title('input image')

delta = 0

layer = slgbuilder.GraphObject(I)
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)

helper.solve()
segmentation = helper.what_segments(layer)
segmentation_line_s0 = segmentation.shape[0] - np.argmax(segmentation[::-1,:], axis=0) - 1

ax[1].imshow(I, cmap='gray')
ax[1].plot(segmentation_line_s0, 'r')
ax[1].set_title(f'delta = {delta}')


#%% a smoother line
delta = 5

layer = slgbuilder.GraphObject(I)
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)

helper.solve()
segmentation = helper.what_segments(layer)
segmentation_line_s5 = segmentation.shape[0] - np.argmax(segmentation[::-1,:], axis=0) - 1

ax[2].imshow(I, cmap='gray')
ax[2].plot(segmentation_line_s5, 'r')
ax[2].set_title(f'delta = {delta}')


layers = [slgbuilder.GraphObject(I), slgbuilder.GraphObject(I)]
delta = 3

helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)  
helper.add_layered_containment(layers[0], layers[1], min_margin=15)

helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]

ax[3].imshow(I, cmap='gray')
for line in segmentation_lines:
    ax[3].plot(line, 'r')
ax[3].set_title('two dark lines')

plt.show()



d = 0
for i in range(I.shape[1]):
    d += abs(segmentation_line_s0[i]-segmentation_line_s5[i])

d = d/I.shape[1]
print(d)
