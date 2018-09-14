from keras import backend as K
from keras import optimizers
from keras.layers import Input, Convolution2D, UpSampling2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape
from keras.models import Model
import numpy as np
import glob
import os

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))



def DeepAuto(image_dim):
    channels = 3
    input_img = Input(shape=image_dim)

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    encoded = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    decoded = Convolution2D(channels, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=[])

    return autoencoder


def DeepAuto2(image_dim):
    channels = 3
    input_img = Input(shape=image_dim)

    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    encoded = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(encoded)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    decoded = Convolution2D(channels, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=[])

    return autoencoder


def SRCNN(image_dim):
    input_img = Input(shape=image_dim)

    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)
    out = Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, out)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=[])

    return autoencoder


def SRCNN1618(image_dim):
    input_img = Input(shape=image_dim)

    x = Convolution2D(512, 16, 16, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(512, 1, 1, activation='relu', border_mode='same')(x)
    out = Convolution2D(3, 8, 8, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, out)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=[])

    return autoencoder


def DeepDenoiseSR(image_dim):
    channels = 3
    n1 = 64
    n2 = 128
    n3 = 256

    input_img = Input(shape=image_dim)
    c1 = Convolution2D(n1, 3, 3, activation='relu', border_mode='same')(input_img)
    c1 = Convolution2D(n1, 3, 3, activation='relu', border_mode='same')(c1)

    x = MaxPooling2D((2, 2))(c1)

    c2 = Convolution2D(n2, 3, 3, activation='relu', border_mode='same')(x)
    c2 = Convolution2D(n2, 3, 3, activation='relu', border_mode='same')(c2)

    x = MaxPooling2D((2, 2))(c2)

    c3 = Convolution2D(n3, 3, 3, activation='relu', border_mode='same')(x)

    x = UpSampling2D()(c3)

    c2_2 = Convolution2D(n2, 3, 3, activation='relu', border_mode='same')(x)
    c2_2 = Convolution2D(n2, 3, 3, activation='relu', border_mode='same')(c2_2)

    m1 = merge([c2, c2_2], mode='sum')
    m1 = UpSampling2D()(m1)

    c1_2 = Convolution2D(n1, 3, 3, activation='relu', border_mode='same')(m1)
    c1_2 = Convolution2D(n1, 3, 3, activation='relu', border_mode='same')(c1_2)

    m2 = merge([c1, c1_2], mode='sum')

    decoded = Convolution2D(channels, 5, 5, activation='sigmoid', border_mode='same')(m2)

    model = Model(input_img, decoded)

    adam = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=[])

    return model


def DeepDenoiseSR2(image_dim):
    channels = 3
    n1 = 128
    n2 = 256
    n3 = 512

    input_img = Input(shape=image_dim)
    d = Dropout(0.5)(input_img)
    c1 = Convolution2D(n1, 3, 3, activation='relu', border_mode='same')(d)
    c1 = Convolution2D(n1, 3, 3, activation='relu', border_mode='same')(c1)

    d = Dropout(0.5)(c1)

    c2 = Convolution2D(n2, 3, 3, activation='relu', border_mode='same')(d)
    c2 = Convolution2D(n2, 3, 3, activation='relu', border_mode='same')(c2)

    c3 = Convolution2D(n3, 3, 3, activation='relu', border_mode='same')(c2)

    c2_2 = Convolution2D(n2, 3, 3, activation='relu', border_mode='same')(c3)
    c2_2 = Convolution2D(n2, 3, 3, activation='relu', border_mode='same')(c2_2)

    m1 = merge([c2, c2_2], mode='sum')
    x = Dropout(0.5)(m1)

    c1_2 = Convolution2D(n1, 3, 3, activation='relu', border_mode='same')(x)
    c1_2 = Convolution2D(n1, 3, 3, activation='relu', border_mode='same')(c1_2)

    m2 = merge([c1, c1_2], mode='sum')
    x = Dropout(0.5)(m2)

    decoded = Convolution2D(channels, 5, 5, activation='sigmoid', border_mode='same')(x)

    model = Model(input_img, decoded)

    adam = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=[])

    return model


def build_model(model_dir, model_type, image_dim):
    model = model_type(image_dim)
    files = glob.glob(model_dir + '/*.hdf5')
    if len(files):
        files.sort(key=os.path.getmtime, reverse=True)
        print('Loading model ' + files[0])
        if os.path.isfile(files[0]):
            model.load_weights(files[0])
    return model
