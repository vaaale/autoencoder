import glob
import os
from keras.callbacks import RemoteMonitor, ModelCheckpoint
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from image_patches import dataset
from progress_monitor import ProgressMonitor

img_width = img_height = 32
img_depth = 3
image_dim = (img_width, img_height, img_depth)

x_train, x_test = dataset(data_dir='data/train', dim=(image_dim[0], image_dim[1]), max_patches=10000)
print(x_train.shape)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), *image_dim))
x_test = np.reshape(x_test, (len(x_test), *image_dim))

noise_factor = 0.25
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


def create_model():
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


def create_model2():
    input_img = Input(shape=image_dim)

    x = Convolution2D(64, 33, 33, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(64, 33, 33, activation='relu', border_mode='same')(x)

    decoded = Convolution2D(img_depth, 32, 32, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)

    return autoencoder


def build_model(model_dir):
    model = create_model()
    files = glob.glob(model_dir+'/*.hdf5')
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
progress = ProgressMonitor(x_test=x_test, x_test_noisy=x_test_noisy, dim=image_dim)
callbacks_list = [checkpoint, remote, progress]

autoencoder.fit(x_train_noisy, x_train,
                nb_epoch=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=callbacks_list)

decoded_imgs = autoencoder.predict(x_test_noisy)

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
    plt.imshow(x_test_noisy[i].reshape(img_width, img_height, img_depth))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(img_width, img_height, img_depth))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('end-result.png', bbox_inches='tight')
plt.show()
