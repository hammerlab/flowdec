

# TZCYXS

import numpy as np
from skimage.external.tifffile import imread, imsave
x = np.ones((8, 5, 4, 200, 100)).astype(np.float32)
imsave('C:/Users/User/test.tif', x, imagej=True) 
y = imread('C:/Users/User/test.tif')
print(x.shape, y.shape)