import glob

import os
#os.environ['THEANO_FLAGS'] = "device=cpu"
#import theano

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from models import DeepDenoiseSR, DeepAuto, DeepDenoiseSR2
from scipy.ndimage.filters import gaussian_filter
import ntpath


p_dim = (32, 32, 3)

# print('Building model...')
# model = DeepDenoiseSR2(p_dim)
# print('Loadin weights...')
# model.load_weights('model/backup/DeepDenoiseSR-0.0006.hdf5')


def remaster(orig, scale, p_dim, model):
    input_image = imresize(orig, 1/2)
    # if scale != 1:
    #     input_image = imresize(orig, scale*100)
    # else:
    #     input_image = orig
    print('Image shape')
    print(input_image.shape)

    patches = []
    m_dim = int(input_image.shape[0] - p_dim[0])
    n_dim = int(input_image.shape[1] - p_dim[1])
    for m in np.arange(0, m_dim, p_dim[0]):
        for n in np.arange(0, n_dim, p_dim[1]):
            patch_x = input_image[m:m + p_dim[1], n:n + p_dim[0]]
            input = patch_x.reshape((1, *p_dim)) / 255.
            patches.append(input)

    arr = np.concatenate(patches, axis=0)
    width = (np.ceil(n_dim/p_dim[1]))
    height = (np.ceil(m_dim/p_dim[0]))
    result = model.predict(arr, batch_size=1024)
    output = np.zeros((height*p_dim[0], width*p_dim[1], p_dim[2]))
    for m in np.arange(height):
        for n in np.arange(width):
            output[m*p_dim[0]:m*p_dim[0] + p_dim[0], n*p_dim[1]:n*p_dim[1] + p_dim[1]] = result[m*width + n]

    return input_image, output

data_dir = '../../data2'
input_files = sorted(glob.glob(data_dir + '/test/*.jpg'))
excludes = sorted(glob.glob(data_dir + '/rem/*.jpg'))
y_files = [file for file in input_files if file not in excludes]


for file in y_files:
    print(ntpath.basename(file))
    orig = mpimg.imread(file)
    dim = orig.shape
    orig = orig.reshape(1, *dim)
    print('Building model...')
    model = DeepDenoiseSR2(dim)
    print('Loadin weights...')
    model.load_weights('model/backup/DeepDenoiseSR-0.0006.hdf5')
    input = orig
    output = model.predict(orig)
    #input, output = remaster(gaussian_filter(orig, sigma=0.7), 1, p_dim)
    #mpimg.imsave(data_dir + '/output/' + ntpath.basename(file), output)


    print('Displaying result.')
    plt.figure(figsize=(20, 4))
    ax = plt.subplot(1, 3, 1)
    plt.imshow(orig)

    ax = plt.subplot(1, 3, 2)
    plt.imshow(input)

    ax = plt.subplot(1, 3, 3)
    plt.imshow(output*1.9)
    plt.show()
