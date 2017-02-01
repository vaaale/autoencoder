import glob
import ntpath
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize

from models import DeepDenoiseSR


def build_model(model_dir, model_type, image_dim):
    model = model_type(image_dim)
    files = glob.glob(model_dir + '/*.hdf5')
    if len(files):
        files.sort(key=os.path.getmtime, reverse=True)
        print('Loading model ' + files[0])
        if os.path.isfile(files[0]):
            model.load_weights(files[0])
    return model


file = 'test.jpg'
print(ntpath.basename(file))
image = (imresize(imread(file, mode='RGB'), (480, 640)) / 255.).astype(np.float32)
print('Image shape: {}'.format(image.shape))

print('Building model')
autoencoder = build_model('model/DeepDenoiseSR', DeepDenoiseSR, image.shape)
autoencoder.summary()

print('Predicting....')

output = autoencoder.predict(image.astype(np.float32).reshape((1, *image.shape)))

output = (output.reshape(image.shape) * 255).astype(np.uint8)

print('Displaying result.')
print(output.shape)
plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 2, 1)
ax.set_title('Original')
plt.imshow(image)

ax = plt.subplot(1, 2, 2)
ax.set_title('Reconstruction')
plt.imshow(output)

plt.show()
