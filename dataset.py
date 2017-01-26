import glob
import scipy.stats as ss
from scipy.misc import imresize
import numpy as np
import matplotlib.image as mpimg
from multiprocessing import Process, Queue


def random_range(length, num):
    x = np.arange(-length/2, length/2, dtype=int)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale=length/5) - ss.norm.cdf(xL, scale=length/5)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size=num, p=prob)
    return nums


def patchify(data_dir='data', dim=(32, 32), max_patches=2000, infinite=False):
    y_files = glob.glob(data_dir + '/*.jpg')
    np.random.shuffle(y_files)
    y_files = y_files[:min(100, len(y_files))]
    print('Loading files...')

    def load(file):
        print(file)
        image = imresize(mpimg.imread(file), (512, 512))
        return image

    # y_images = [load(file) / 255. for file in y_files]
    y_images = [imresize(mpimg.imread(file), (512, 512)) / 255. for file in y_files]
    x_images = [pixelfy(gen_noise(image)) for image in y_images]
    images = list(zip(x_images, y_images))
    print('Generating.... ' + data_dir)
    while True:
        np.random.shuffle(images)
        for x_image, y_image in images:
            m_dim = int(x_image.shape[0] - dim[0])
            n_dim = int(x_image.shape[1] - dim[1])
            for _ in np.arange(max_patches):
                np.random.normal()
                # m = np.random.randint(0, m_dim)
                # n = np.random.randint(0, n_dim)
                m = int(m_dim / 2) + random_range(m_dim, 1)[0]
                n = int(n_dim / 2) + random_range(n_dim, 1)[0]
                patch_x = x_image[m:m + dim[1], n:n + dim[0]]
                patch_y = y_image[m:m + dim[1], n:n + dim[0]]

                yield patch_x, patch_y
        if not infinite:
            break


def gen_noise(x_image):
    width, height, channels = x_image.shape
    x_image = x_image[:, :, 0:channels] * np.asarray(np.random.rand(height, width, 1) > 0.07, dtype='float32')
    np.place(x_image, x_image == 0., np.random.random_sample())
    return x_image


def pixelfy(x_image):
    cell_size = 3
    height, width, channels = x_image.shape
    for m in np.arange(0, height, cell_size):
        for n in np.arange(0, width, cell_size):
            x_image[m:m + cell_size, n:n + cell_size] = x_image[m:m + cell_size, n:n + cell_size].mean(axis=(0, 1))
    return x_image


def producer(p_idx, data_dir, dim, max_patches, q):
    generator = patchify(data_dir=data_dir, dim=dim, max_patches=max_patches, infinite=True)
    while True:
        x_image, y_image = next(generator)
        q.put((x_image, y_image))


def stream_patches_live(data_dir='../../data2/patches', dim=(32, 32), batch_size=128, max_patches=1000, nb_workers=1):
    # To get rid of linter warning
    generator = None
    q = None
    if nb_workers > 1:
        q = Queue(2000)
        [Process(target=producer, args=(p_idx, data_dir, dim, max_patches, q)).start() for p_idx in np.arange(nb_workers)]
    else:
        generator = patchify(data_dir=data_dir, dim=dim, max_patches=max_patches, infinite=True)

    while True:
        batch_x = []
        batch_y = []
        for _ in np.arange(batch_size):
            if nb_workers > 1:
                x_image, y_image = q.get()
            else:
                x_image, y_image = next(generator)
            batch_x.append(x_image)
            batch_y.append(y_image)
        batch_x = np.asarray(batch_x).astype('float32')
        batch_y = np.asarray(batch_y).astype('float32')

        yield batch_x, batch_y


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    batch_size = 128
    gen = stream_patches_live(data_dir='../../Pictures/people/test', batch_size=batch_size, dim=(64, 64), max_patches=1000)

    n = 10
    for x_train, y_train in gen:
        x_train = x_train[0:n,...]
        y_train = y_train[0:n,...]
        plt.figure(figsize=(20, 4))
        for i in range(n):
            dim_x = x_train[i].shape
            dim_y = y_train[i].shape
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(x_train[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display noisy
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(y_train[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
