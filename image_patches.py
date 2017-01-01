import glob
import os
from scipy.misc import imresize
import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage.filters import gaussian_filter


def patches(data_dir='data', dim=(32, 32), scale=2, max_patches=2000, infinite=False):
    lr_patch_dim = (int(dim[0] / scale), int(dim[1] / scale))
    y_files = glob.glob(data_dir + '/*.jpg')
    print('Generating....')
    while True:
        np.random.shuffle(y_files)
        for y_file in y_files:
            print('Reading patches from ' + y_file)
            y_image = mpimg.imread(y_file)
            y_image = imresize(y_image, (256, 256))
            x_image = y_image

            m_dim = int(x_image.shape[0] - dim[0])
            n_dim = int(x_image.shape[1] - dim[1])
            for _ in np.arange(max_patches):
                m = np.random.randint(0, m_dim)
                n = np.random.randint(0, n_dim)
                patch_x = x_image[m:m + dim[1], n:n + dim[0]]
                patch_y = patch_x
                patch_x = gaussian_filter(patch_x, sigma=0.5)
                patch_x = imresize(patch_x, lr_patch_dim, interp='bicubic')
                patch_x = imresize(patch_x, dim, interp='bicubic')
                #patch_x = patch_x + 0.2 * patch_x*np.random.normal(loc=0.0, scale=0.5, size=patch_x.shape)

                yield patch_x, patch_y
        if not infinite:
            break


def batch_generator(data_dir='data', dim=(28, 28), batch_size=128, max_patches=2000):
    patch_generator = patches(data_dir, dim=dim, max_patches=max_patches)
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


def build_dataset(data_dir='data', target_dir='../../data2/patches', dim=(32, 32), max_patches=2000):
    generator = patches(data_dir=data_dir, dim=dim, max_patches=max_patches)
    i = 0
    for x, y in generator:
        mpimg.imsave(target_dir + '/x/patch_{:d}.jpg'.format(i), x)
        mpimg.imsave(target_dir + '/y/patch_{:d}.jpg'.format(i), y)
        i += 1


def stream_patches(data_dir='../../data2/patches', batch_size=128):
    file_names = [f for f in sorted(os.listdir(data_dir + "/x/"))]

    batch_x = []
    batch_y = []
    while True:
        np.random.shuffle(file_names)
        for file in file_names:
            x_image = mpimg.imread(data_dir + '/x/' + file) / 255.
            y_image = mpimg.imread(data_dir + '/y/' + file) / 255.
            batch_x.append(x_image)
            batch_y.append(y_image)
            if len(batch_x) == batch_size:
                batch_x = np.asarray(batch_x).astype('float32')
                batch_y = np.asarray(batch_y).astype('float32')

                yield batch_x, batch_y
                batch_x = []
                batch_y = []


if __name__ == '__main__':
    build_dataset(data_dir='../../data2/train', target_dir='../../data2/patches/train', max_patches=500)
    build_dataset(data_dir='../../data2/test', target_dir='../../data2/patches/test', max_patches=500)
    #stream_patches()

    import matplotlib.pyplot as plt
    batch_size = 10
    #gen = batch_generator(data_dir='z:\\Transport\\images\\train', dim=(32, 32), max_patches=3, batch_size=batch_size)
    gen = stream_patches(data_dir='../../data2/patches/test', batch_size=batch_size)
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

