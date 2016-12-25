import glob
import os

import matplotlib.pyplot as plt
from keras.callbacks import RemoteMonitor, ModelCheckpoint
from keras.layers import Input, Convolution2D, UpSampling2D
from keras.models import Model

from image_patches import batch_generator
from progress_monitor import ProgressMonitor

img_width = img_height = 64
img_depth = 3
image_dim = (img_width, img_height, img_depth)
scale = 1
batch_size = 128


def DeepAuto():
    input_img = Input(shape=image_dim)

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    encoded = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(encoded)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    decoded = Convolution2D(img_depth, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input=input_img, output=encoded)

    return autoencoder


def SRCNN():
    input_img = Input(shape=image_dim)

    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)
    out = Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, out)

    return autoencoder


def build_model(model_dir):
    model = SRCNN()
    files = glob.glob(model_dir + '/*.hdf5')
    if len(files):
        files.sort(key=os.path.getmtime, reverse=True)
        print('Loading model ' + files[0])
        if os.path.isfile(files[0]):
            print('Resuming training')
            model.load_weights(files[0])
    return model


autoencoder = build_model('model')

autoencoder.compile(optimizer='adam', loss='mse')

filepath = "model/model-epoch-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
remote = RemoteMonitor(root='http://localhost:9000')

test_generator = batch_generator(data_dir='data/test', dim=(img_width, img_height), scale=scale, max_patches=500, batch_size=128)

progress = ProgressMonitor(generator=test_generator, dim=image_dim)
callbacks_list = [checkpoint, remote, progress]

generator = batch_generator(data_dir='data/train', dim=(img_width, img_height), scale=scale, max_patches=5000, batch_size=batch_size)
autoencoder.fit_generator(generator, samples_per_epoch=batch_size*500, nb_epoch=30, callbacks=callbacks_list)

x_test, y_test = next(test_generator)
decoded_imgs = autoencoder.predict(x_test)

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
    plt.imshow(y_test[i].reshape(img_width*scale, img_height*scale, img_depth))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[i].reshape(img_width*scale, img_height*scale, img_depth))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('end-result.png', bbox_inches='tight')
plt.show()
