import glob
import os

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard

from dataset import stream_patches_live
from models import SRCNN, SRCNN1618, DeepAuto, DeepDenoiseSR
from progress_monitor import ProgressMonitor

img_width = img_height = 64
img_depth = 3
image_dim = (img_width, img_height, img_depth)
batch_size = 128


def build_model(model_dir, model_type):
    model = model_type(image_dim)
    files = glob.glob(model_dir + '/*.hdf5')
    if len(files):
        files.sort(key=os.path.getmtime, reverse=True)
        print('Loading model ' + files[0])
        if os.path.isfile(files[0]):
            print('Resuming training')
            model.load_weights(files[0])
    return model


if __name__ == '__main__':
    autoencoder = build_model('model', SRCNN)
    autoencoder.summary()
    generator = stream_patches_live(data_dir='../../Pictures/people/train', dim=(64, 64), batch_size=batch_size, max_patches=batch_size)
    val_generator = stream_patches_live(data_dir='../../Pictures/people/test', dim=(64, 64), batch_size=batch_size, max_patches=batch_size)
    test_generator = stream_patches_live(data_dir='../../Pictures/people/test', dim=(64, 64), batch_size=batch_size, max_patches=1)

    progress = ProgressMonitor(generator=test_generator, dim=image_dim)
    tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint('model/model-{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True,
                                 mode='min', save_weights_only=True, verbose=1)

    callbacks_list = [checkpoint, progress, tb]

    hist = autoencoder.fit_generator(generator, samples_per_epoch=batch_size * 500, nb_epoch=200, callbacks=callbacks_list,
                                     validation_data=val_generator, nb_val_samples=batch_size * 100, verbose=1,
                                     nb_worker=1)

    x_test, y_test = next(test_generator)
    decoded_imgs = autoencoder.predict(x_test)

    scale = 1
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(img_width, img_height, img_depth))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(y_test[i].reshape(img_width * scale, img_height * scale, img_depth))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(decoded_imgs[i].reshape(img_width * scale, img_height * scale, img_depth))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('end-result.png', bbox_inches='tight')
    plt.show()
