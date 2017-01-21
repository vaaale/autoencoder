from image_patches import stream_patches_live
import numpy as np

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
        return x_image




    import matplotlib.pyplot as plt

    batch_size = 10
    gen = stream_patches_live(data_dir='../../Pictures/people/test', batch_size=batch_size, dim=(32, 32),
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
