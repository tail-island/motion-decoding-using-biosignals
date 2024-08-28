import numpy as np

from scipy.io import loadmat


mat = loadmat('../input/motion-decoding-using-biosignals/reference.mat')

x = np.sum(np.sum(mat['0005'][0][0][1], axis=2), axis=0)
y = np.sum(np.sum(mat['0005'][0][0][3], axis=2), axis=0)

print(y / x)
