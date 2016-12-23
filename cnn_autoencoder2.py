import glob
import os

import matplotlib.pyplot as plt
from keras.callbacks import RemoteMonitor, ModelCheckpoint
from keras.layers import Input, Convolution2D, UpSampling2D
from keras.models import Model

from image_patches import batch_generator
from progress_monitor import ProgressMonitor

img_width = img_height = 32
img_depth = 3
image_dim = (img_width, img_height, img_depth)
scale = 1

# x_train, x_test = dataset(dim=(image_dim[0], image_dim[1]), max_patches=50)
# print(x_train.shape)
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), *image_dim))
# x_test = np.reshape(x_test, (len(x_test), *image_dim))
#
# noise_factor = 0.2
# nose_channel = 0
# noise_mean = 0.0
# noise_scale = 0.1
# x_train_noisy = x_train.copy()
# x_test_noisy = x_test.copy()
# x_train_noisy[:, :, :, nose_channel] += np.random.normal(loc=noise_mean, scale=noise_scale, size=(len(x_train_noisy), img_width, img_height))
# x_test_noisy[:, :, :, nose_channel] += np.random.normal(loc=noise_mean, scale=noise_scale, size=(len(x_test_noisy), img_width, img_height))
#
# print(x_train_noisy.shape)
# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)
#
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display noisy
#     ax = plt.subplot(1, n, i + 1)
#     plt.imshow(x_test_noisy[i].reshape(img_width, img_height, img_depth))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
#

def SRCNN(x_dim, scale):
    input_img = Input(shape=x_dim)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    encoded = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D(size=(scale, scale))(x)
    decoded = Convolution2D(img_depth, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    #encoder = Model(input=input_img, output=encoded)

    return autoencoder


def DeepAuto():
    input_img = Input(shape=image_dim)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    encoded = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    decoded = Convolution2D(img_depth, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input=input_img, output=encoded)

    return autoencoder


def build_model(model_dir):
    model = DeepAuto()
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

batch_size = 128
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
