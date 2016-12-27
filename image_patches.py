import glob

from scipy.misc import imresize
from sklearn.feature_extraction import image
import numpy as np
import matplotlib.image as mpimg
from skimage.filters import gaussian
from scipy.ndimage.filters import gaussian_filter


def patches(data_dir='data', dim=(32, 32), scale=2, max_patches=2000):
    dim_y = (dim[0] * scale, dim[1] * scale)
    y_files = glob.glob(data_dir + '/*.jpg')
    print('Generating....')
    while True:
        np.random.shuffle(y_files)
        for y_file in y_files:
            #print('Reading patches from ' + y_file)
            y_image = mpimg.imread(y_file)
            y_shape = y_image.shape
            if scale > 1:
                x_image = imresize(y_image, (int(y_shape[0]/scale), int(y_shape[1]/scale), int(y_shape[2])))
            else:
                x_image = y_image

            # Apply Gaussian Blur to Y
            x_image = gaussian_filter(x_image, sigma=0.5)
            x_image = imresize(x_image, 1/2, interp='bicubic')
            x_image = imresize(x_image, 200, interp='bicubic')


            #x_image = imresize(x_image, 1 / 3)
            #x_image = imresize(x_image, 300)

            # noise_factor = 0.15
            # batch_x_noisy = batch_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)
            x_image = x_image / 255.
            y_image = y_image / 255.

            m_dim = int(x_image.shape[0] - dim[0])
            n_dim = int(x_image.shape[1] - dim[1])
            for _ in np.arange(max_patches):
                m = np.random.randint(0, m_dim)
                n = np.random.randint(0, n_dim)
                patch_x = x_image[m:m + dim[1], n:n + dim[0]]
                patch_y = y_image[m*scale:m*scale + dim_y[1], n*scale:n*scale + dim_y[0]]
                yield patch_x, patch_y


def batch_generator(data_dir='data', dim=(28, 28), scale=2, batch_size=128, max_patches=2000):
    patch_generator = patches(data_dir, dim=dim, scale=scale, max_patches=max_patches)
    batch_x = []
    batch_y = []
    for x, y in patch_generator:
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) == batch_size:
            batch_x = np.asarray(batch_x).astype('float32')
            batch_y = np.asarray(batch_y).astype('float32')

            yield batch_x, batch_y
            batch_x = []
            batch_y = []


def dataset(data_dir='data', dim=(28, 28), test_size=0.3, max_patches=10000):
    data = []
    files = glob.glob(data_dir + '/*.jpg')
    for file_name in files:
        one_image = mpimg.imread(file_name)
        patches = image.extract_patches_2d(one_image, dim, max_patches=max_patches)
        data.append(patches[:])

    data = np.concatenate(data, axis=0)
    split = int(len(data) * (1 - test_size))

    x_train = data[:split]
    x_test = data[split:]
    return x_train, x_test


def dataset_noisy(data_dir='data', dim=(28, 28), test_size=0.3, max_patches=10000):
    data = []
    files = glob.glob(data_dir + '/*.jpg')
    for file_name in files:
        one_image = mpimg.imread(file_name)
        patches = image.extract_patches_2d(one_image, dim, max_patches=max_patches)
        data.append(patches[:])

    data = np.concatenate(data, axis=0)
    split = int(len(data) * (1 - test_size))

    x_train = data[:split]
    x_test = data[split:]

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
    x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

    noise_factor = 0.15
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return x_train_noisy, x_test_noisy


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    batch_size = 10
    gen = batch_generator(data_dir='z:\\Transport\\images\\train', dim=(64, 64), scale=4, max_patches=3, batch_size=batch_size)
    for x_train, y_train in gen:
        plt.figure(figsize=(20, 4))
        for i in range(batch_size):
            dim_x = x_train[i].shape
            dim_y = y_train[i].shape
            # display original
            ax = plt.subplot(3, batch_size, i + 1)
            plt.imshow(x_train[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display noisy
            ax = plt.subplot(3, batch_size, i + 1 + batch_size)
            plt.imshow(y_train[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

