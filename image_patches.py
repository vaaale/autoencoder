import glob
import os
import threading

from scipy.misc import imresize
import numpy as np
import matplotlib.image as mpimg


def patchify(data_dir='data', dim=(32, 32), scale=2, max_patches=2000, infinite=False):
    lr_patch_dim = (int(dim[0] / scale), int(dim[1] / scale))
    y_files = glob.glob(data_dir + '/*.jpg')
    print('Generating.... ' + data_dir)
    while True:
        np.random.shuffle(y_files)
        for y_file in y_files:
            # print('Reading patches from ' + y_file)
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
                # patch_x = imresize(patch_x, lr_patch_dim, interp='bicubic')
                # patch_x = imresize(patch_x, dim, interp='bicubic')

                yield patch_x, patch_y
        if not infinite:
            break


def batch_generator(data_dir='data', dim=(28, 28), batch_size=128, max_patches=2000):
    patch_generator = patchify(data_dir, dim=dim, max_patches=max_patches)
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


def build_dataset(data_dir='data', dim=(32, 32), max_patches=2000, rebuild=False):
    if not os.path.isdir(data_dir + '/patches/x'):
        os.makedirs(data_dir + '/patches/x')
        rebuild = True
    if not os.path.isdir(data_dir + '/patches/y'):
        os.makedirs(data_dir + '/patches/y')
        rebuild = True

    if rebuild:
        generator = patchify(data_dir=data_dir, dim=dim, max_patches=max_patches)
        i = 0
        for x, y in generator:
            mpimg.imsave(data_dir + '/patches/x/patch_{:d}.jpg'.format(i), x)
            mpimg.imsave(data_dir + '/patches/y/patch_{:d}.jpg'.format(i), y)
            i += 1

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def stream_patches_live(data_dir='../../data2/patches', dim=(32, 32), batch_size=128, max_patches=500, scale=2,
                        noise_fn=None):

    batch_x = []
    batch_y = []
    while 1:
        generator = patchify(data_dir=data_dir, dim=dim, scale=scale, max_patches=max_patches, infinite=True)
        import time
        for x_image, y_image in generator:
            # start = time.time()
            x_image = x_image / 255.
            y_image = y_image / 255.
            if noise_fn is not None:
                x_image = noise_fn(x_image)
            batch_x.append(x_image)
            batch_y.append(y_image)
            if len(batch_x) == batch_size:
                # with ThreadPoolExecutor(4) as pool:
                #     batch_x = [x for x in pool.map(noise_fn, batch_x)]
                batch_x = np.asarray(batch_x).astype('float32')
                batch_y = np.asarray(batch_y).astype('float32')

                # end = time.time()
                # print('Batch generated in: ' + str(end - start))
                yield batch_x, batch_y
                batch_x = []
                batch_y = []


def stream_patches(data_dir='../../data2/patches', batch_size=128, noise_fn=None):
    full_path = data_dir + "/patches/x/"
    if not os.path.isdir(full_path):
        build_dataset(data_dir=data_dir, max_patches=500)

    file_names = [f for f in sorted(os.listdir(full_path))]

    batch_x = []
    batch_y = []
    while True:
        np.random.shuffle(file_names)
        for file in file_names:
            x_image = mpimg.imread(data_dir + '/patches/x/' + file) / 255.
            y_image = mpimg.imread(data_dir + '/patches/y/' + file) / 255.
            x_image = noise_fn(x_image)
            batch_x.append(x_image)
            batch_y.append(y_image)
            if len(batch_x) == batch_size:
                batch_x = np.asarray(batch_x).astype('float32')
                batch_y = np.asarray(batch_y).astype('float32')

                yield batch_x, batch_y
                batch_x = []
                batch_y = []


if __name__ == '__main__':
    def gen_noise(x_image):
        width, height, channels = x_image.shape
        x_image = x_image[:, :, 0:channels] * np.asarray(np.random.rand(height, width, 1) > 0.07, dtype='float32')
        np.place(x_image, x_image == 0., np.random.random_sample())
        return x_image


    def pixelfy(x_image):
        cell_size = 2
        height, width, channels = x_image.shape
        for m in np.arange(0, height, cell_size):
            for n in np.arange(0, width, cell_size):
                x_image[m:m + cell_size, n:n + cell_size, 0] = x_image[m:m + cell_size, n:n + cell_size, 0].mean()
                x_image[m:m + cell_size, n:n + cell_size, 1] = x_image[m:m + cell_size, n:n + cell_size, 1].mean()
                x_image[m:m + cell_size, n:n + cell_size, 2] = x_image[m:m + cell_size, n:n + cell_size, 2].mean()
        return gen_noise(x_image)


    import matplotlib.pyplot as plt

    batch_size = 10
    gen = stream_patches_live(data_dir='../../Pictures/people/test', batch_size=10, dim=(32, 32),
                              max_patches=500, noise_fn=pixelfy)
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
