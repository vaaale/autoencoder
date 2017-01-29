from keras.callbacks import Callback
import matplotlib.pyplot as plt


class TrainingHistory(Callback):
    def __init__(self, y=None, x=None, generator=None):
        super().__init__()
        self.x = x
        self.y = y
        self.generator = generator

    def on_train_begin(self, logs={}):
        self.losses = []
        self.predictions = []
        self.i = 0
        self.save_every = 50

    def on_batch_end(self, batch, logs={}):
        if self.generator is not None:
            x, y = next(self.generator)
        else:
            x = self.x

        self.losses.append(logs.get('loss'))
        self.i += 1
        if self.i % self.save_every == 0:
            pred = self.model.predict(x)
            self.predictions.append(pred)


class ProgressMonitor(Callback):

    def __init__(self, y=None, x=None, generator=None, path='.', dim=(32, 32, 3)):
        super().__init__()
        self.x_test = y
        self.x_test_noisy = x
        self.generator = generator
        self.dim = dim
        self.path = path
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

        filename = self.path + '/epoch-{epoch:02d}-{val_loss:.6f}.png'.format(epoch=epoch, **logs)
        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(y[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display noisy
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(x[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            #display reconstruction
            ax = plt.subplot(3, n, i + 1 + 2 * n)
            plt.imshow(decoded_imgs[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(filename, bbox_inches='tight')

