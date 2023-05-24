import numpy as np
import skimage.io
import matplotlib.pyplot as plt



img = skimage.io.imread("/home/christian/advancedImageAnalysis/Exam/2021/EXAM_DATA_2021/frame.png")/255

plt.figure()
plt.imshow(img)


print(img.shape)
point = np.array([img.shape[1]/2-40,img.shape[0]/2+40])

plt.scatter(point[0],point[1])


I_total = np.sum(img)
I_inside = np.sum(img[60:140,35:115])
I_outside = I_total-I_inside

m_in = I_inside/((140-60)*(115-35))
m_out = I_outside/((img.shape[0]*img.shape[1])-((140-60)*(115-35)))

F_ext = (m_in-m_out)*(2*img[int(point[0]),int(point[1])]-m_in-m_out)
print(2*img[int(point[0]),int(point[1])])
print(F_ext)
plt.show()