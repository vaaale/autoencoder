import matplotlib.pyplot as plt
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D
from scipy.misc import imresize
from skimage.filters import gaussian
import numpy as np
import matplotlib.image as mpimg

dim = (32,32, 3)

def SRCNN():
    input_img = Input(shape=dim)

    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)
    out = Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, out)

    return autoencoder

print('Building model...')
model = SRCNN()
print('Loadin weights...')
model.load_weights('model/backup/SRCNN-64x64-27-0.0003.hdf5')

face = mpimg.imread('data/train/img3.jpg')


shrunk = imresize(face, 1/3)

print('Restoring image....')
output = np.zeros(shrunk.shape)
m_dim = int(shrunk.shape[0] - dim[0])
n_dim = int(shrunk.shape[1] - dim[1])
for m in np.arange(0, m_dim, dim[0]):
    for n in np.arange(0, n_dim, dim[1]):
        patch_x = shrunk[m:m + dim[1], n:n + dim[0]]
        input = patch_x.reshape((1, *dim)) / 255.
        recons = model.predict(input)

        output[m:m + dim[1], n:n + dim[0]] = recons

print('Displaying result.')

plt.figure(figsize=(20, 4))
ax = plt.subplot(1, 3, 1)
plt.imshow(face)
ax = plt.subplot(1, 3, 2)
plt.imshow(shrunk)
ax = plt.subplot(1, 3, 3)
plt.imshow(output)
plt.show()