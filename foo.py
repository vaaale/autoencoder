from tensorflow.python.keras import Input, Model, optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Conv2D, Add, Dropout, concatenate, UpSampling2D, Dense
from tensorflow.contrib.keras import applications
import tensorflow as tf
from progress_monitor import ProgressMonitor
from tensorboard import TensorBoard

from datagenerator import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt


img_width = 224
img_height = 224
batch_size = 16
image_dim = [img_width, img_height, 3]




def DeepDenoiseSR2(image_dim, loss=None):
    n1 = 64
    n2 = 128
    n3 = 256

    input_img = Input(shape=image_dim)
    c1 = Convolution2D(n1, 3, strides=(1, 1), activation='relu', padding='same')(input_img)
    c1 = Convolution2D(n1, 3, strides=(1, 1), activation='relu', padding='same')(c1)

    c2 = Convolution2D(n2, 3, strides=(1, 1), activation='relu', padding='same')(c1)
    c2 = Convolution2D(n2, 3, strides=(1, 1), activation='relu', padding='same')(c2)

    c3 = Convolution2D(n3, 3, strides=(1, 1), activation='relu', padding='same')(c2)

    c2_2 = Convolution2D(n2, 3, strides=(1, 1), activation='relu', padding='same')(c3)
    c2_2 = Convolution2D(n2, 3, strides=(1, 1), activation='relu', padding='same')(c2_2)

    m1 = Add()([c2, c2_2])

    c1_2 = Convolution2D(n1, 3, strides=(1, 1), activation='relu', padding='same')(m1)
    c1_2 = Convolution2D(n1, 3, strides=(1, 1), activation='relu', padding='same')(c1_2)

    m2 = Add()([c1, c1_2])

    decoded = Convolution2D(image_dim[-1], 5, strides=(1, 1), activation='sigmoid', padding='same')(m2)

    model = Model(input_img, decoded)

    # adam = optimizers.Adam(lr=1e-3)
    # model.compile(optimizer=adam, loss=loss, metrics=[])

    return model

main_model = DeepDenoiseSR2(image_dim)

# input1 = Input(shape=image_dim)
main_input = main_model.input


loss_base = applications.VGG16(include_top=False, weights='imagenet')
loss_base = Model(loss_base.input, loss_base.layers[-10].output)

base_out = loss_base(main_model.output)

model = Model(inputs=[main_input], outputs=[base_out])

model.summary()

print('Hurra!')