from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
from scipy.misc import imresize
from skimage.filters import gaussian
import numpy as np
import matplotlib.image as mpimg
from matplotlib import colors


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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

face = np.asarray(face).astype('float32') / 255.
shrunk = imresize(face, 1/3)
shrunk = gaussian(shrunk, sigma=(0.2, 0.2, 0.7), multichannel=True)
enlarged = imresize(shrunk, 300)

print(face.shape)

hsv = colors.rgb_to_hsv(face)
print(hsv.shape)


plt.figure()
ax = plt.subplot(2, 2, 1)
plt.imshow(face)
ax = plt.subplot(2, 2, 2)
plt.imshow(enlarged)
plt.show()
