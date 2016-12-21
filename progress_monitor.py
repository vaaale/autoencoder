from keras.callbacks import Callback
import matplotlib.pyplot as plt


class ProgressMonitor(Callback):

    def __init__(self, x_test, x_test_noisy, dim=(32, 32, 3)):
        super().__init__()
        self.x_test = x_test
        self.x_test_noisy = x_test_noisy
        self.dim = dim
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
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
            plt.imshow(self.x_test[i].reshape(img_width, img_height, img_depth))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display noisy
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(self.x_test_noisy[i].reshape(img_width, img_height, img_depth))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + 2 * n)
            plt.imshow(decoded_imgs[i].reshape(img_width, img_height, img_depth))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(filename, bbox_inches='tight')

