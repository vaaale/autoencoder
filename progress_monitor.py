from keras.callbacks import Callback
import matplotlib.pyplot as plt


class ProgressMonitor(Callback):

    def __init__(self, x_test=None, x_test_noisy=None, generator=None, dim=(32, 32, 3)):
        super().__init__()
        self.x_test = x_test
        self.x_test_noisy = x_test_noisy
        self.generator = generator
        self.dim = dim
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if self.generator is not None:
            x, y = next(self.generator)
            decoded_imgs = self.model.predict(x)
        else:
            x = self.x_test_noisy
            y = self.x_test
            decoded_imgs = self.model.predict(self.x_test_noisy)
        img_width = self.dim[0]
        img_height = self.dim[1]
        img_depth = self.dim[2]
        filename = 'epoch-{epoch:02d}-{loss:.4f}.png'.format(epoch=epoch, **logs)
        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(x[i].reshape(img_width, img_height, img_depth))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display noisy
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(y[i].reshape(img_width*2, img_height*2, img_depth))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            #display reconstruction
            ax = plt.subplot(3, n, i + 1 + 2 * n)
            plt.imshow(decoded_imgs[i].reshape(img_width*2, img_height*2, img_depth))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(filename, bbox_inches='tight')

