import numpy as np
import snake_functions
import imageio
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import skimage.draw
from scipy import linalg

def makeNormals(snake):
    N = snake.shape[1]
    normal = np.zeros([2,N])
    for i in range(N):
        normal[:,i] = np.array([(snake[1,(i+1)%N]-snake[1,(i-1)%N])*(-1),(snake[0,(i+1)%N]-snake[0,(i-1)%N])*(-1)])
    return normal

def curveSmoothImplicit(curve,alpha, beta):
    N = curve.shape[0]
    A = np.diag(-2*np.ones(N))+np.diag(np.ones(N-1),k=-1)+np.diag(np.ones(N-1),k=1)
    A[0,N-1] = 1
    A[N-1,0] = 1
    B = np.diag(-6*np.ones(N))+np.diag(4*np.ones(N-1),k=-1)+np.diag(4*np.ones(N-1),k=1)+np.diag(-np.ones(N-2),k=-2)+np.diag(-np.ones(N-2),k=2)
    B[0,N-2] = -1
    B[N-2,0] = -1
    B[N-1,1] = -1
    B[1,N-1] = -1
    B[0,N-1] = 4
    B[N-1,0] = 4
    X = linalg.inv((np.eye(N)-alpha*A-beta*B)) @ curve
    return X

def updateStep(Im,snake):
    mask = skimage.draw.polygon2mask(Im.shape,snake.T).T

    mean_inside = np.sum(np.sum(mask*Im)) / np.sum(np.sum(mask))
    mean_outside = np.sum(np.sum((1-mask)*Im)) / np.sum(np.sum(1-mask))
    magnitude = 6* (mean_inside - mean_outside) * (2*Im[snake[0,:].astype(int),snake[1,:].astype(int)]-mean_inside-mean_outside)
    print(magnitude[0])
    normals = makeNormals(snake)
    print(normals[:,0])
    newSnake = snake + magnitude*normals
    newSnake = curveSmoothImplicit(newSnake.T,3,2).T
    return newSnake


vidoe = imageio.get_reader("crawling_amoeba.mov")

Im = skimage.io.imread("plusplus.png")
Im = skimage.color.rgb2gray(Im)

plt.figure()
plt.imshow(Im)

snake_size = 100
snake_center = np.array([int(Im.shape[0]/2),int(Im.shape[1]/2)])
snake_radius = Im.shape[1]/6
print(snake_radius)
print(snake_center)
snake = np.zeros([2,snake_size])

for i in range(snake_size):
    alpha = i * 2 * np.pi/snake_size
    snake[:,i] = (snake_center + snake_radius * np.array([np.cos(alpha),np.sin(alpha)]))

plt.plot(snake[0,:],snake[1,:])
plt.show()

for n in range(100):
    snake = updateStep(Im,snake)
    snake = snake_functions.distribute_points(snake)
    snake = snake_functions.remove_intersections(snake)
    plt.figure(n)
    plt.imshow(Im)
    plt.plot(snake[0,:],snake[1,:])
    plt.show(block=False)
    plt.pause(0.3)
    plt.close()










plt.show()
vidoe.close()