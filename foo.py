import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize


img = mpimg.imread('../../data2/rem/img1.jpg')

large = imresize(img, 10*100, interp='bicubic')

mpimg.imsave('../../data2/rem/large.jpg', large)

