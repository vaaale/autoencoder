from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
from scipy.misc import imresize
from skimage.filters import gaussian
import numpy as np
import matplotlib.image as mpimg


def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()
        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= (255.0 / (maxval - minval))
    return arr


face = mpimg.imread('data/train/img3.jpg')

print(face.shape)

face = np.asarray(face).astype('float32') / 255.
shrunk = imresize(face, 1/3)
shrunk = gaussian(shrunk, sigma=(0.5, 0.5, 0), multichannel=True)
enlarged = imresize(shrunk, 300)

print(face.shape)

plt.figure()
ax = plt.subplot(2, 2, 1)
plt.imshow(face)
ax = plt.subplot(2, 2, 2)
plt.imshow(enlarged)
plt.show()
