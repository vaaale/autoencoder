import glob
from sklearn.feature_extraction import image
import numpy as np
import matplotlib.image as mpimg


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

    noise_factor = 0.25
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return x_train_noisy, x_test_noisy


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x_train, x_test = dataset_noisy(dim=(32, 32), max_patches=5000)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
