import glob


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from models import DeepDenoiseSR, DeepAuto, DeepDenoiseSR2
from scipy.ndimage.filters import gaussian_filter
import ntpath
from itertools import product
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from scipy.misc import imread
import os


def build_model(model_dir, model_type, image_dim):
    model = model_type(image_dim)
    files = glob.glob(model_dir + '/*.hdf5')
    if len(files):
        files.sort(key=os.path.getmtime, reverse=True)
        print('Loading model ' + files[0])
        if os.path.isfile(files[0]):
            model.load_weights(files[0])
    return model


def patch_generator(patches, batch_size):
    batch = []
    for patch in patches:
        batch.append(patch / 255.)
        if len(batch) == batch_size:
            yield np.asarray(batch).astype('float32')
            batch = []




patch_size = 64
print('Building model')
autoencoder = build_model('model', DeepDenoiseSR, (patch_size, patch_size, 3))
autoencoder.summary()

file = 'test.jpg'
print(ntpath.basename(file))
image = imread(file, mode='RGB')
print('Image shape: {}'.format(image.shape))
xx = imresize(image, 300)
patches = extract_patches_2d(xx, (patch_size, patch_size))
print('Number of patches extracted: {}'.format(len(patches)))

print('Predicting....')
predicted_output = []
for x in patch_generator(patches, 512):
    y = autoencoder.predict_on_batch(x)
    y = y * 255
    y = y.astype('uint8')
    predicted_output.append(y)

predictions = np.concatenate(predicted_output)


print('Reconstructing...')
output = reconstruct_from_patches_2d(predictions, xx.shape)

output = output.astype('uint8')

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
