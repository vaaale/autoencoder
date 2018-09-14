from tensorflow.python.keras import Input, Model, optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Conv2D, Add, Dropout, concatenate, UpSampling2D

from progress_monitor import ProgressMonitor
from tensorboard import TensorBoard


from datagenerator import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt

train_data_dir = '/home/alex/Datasets/coco/distorted'
train_label_dir = '/home/alex/Datasets/coco/orig'

val_data_dir = '/home/alex/Datasets/coco/distorted'
val_label_dir = '/home/alex/Datasets/coco/orig'


img_width = 224
img_height = 224
batch_size = 16

datagen = ImageDataGenerator(rescale=1. / 255)

print('Initializing training generator....')
train_gen = datagen.flow_from_directory(
        train_data_dir, train_label_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='input',
        shuffle=True)

print('Initializing validation generator....')
val_gen = datagen.flow_from_directory(
        val_data_dir, val_label_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='input',
        shuffle=True)

print('Initializing tensorboard generator....')
tb_gen = datagen.flow_from_directory(
        val_data_dir, val_label_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle='input')


def unet(pretrained_weights=None, input_size=(224, 224, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv9)

    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def DeepDenoiseSR2(image_dim):
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

    adam = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=[])

    return model


image_dim = [img_width, img_height, 3]
# model = DeepDenoiseSR2(image_dim)
model = unet(None, input_size=image_dim)
# model.summary()

tb = TensorBoard(log_dir='logs/unjpeg', histogram_freq=0, write_graph=True, write_images=True,
                 validation_data=tb_gen)
checkpoint = ModelCheckpoint('model/unjpeg/model-{epoch:02d}-{val_loss:.6f}.hdf5', monitor='val_loss',
                             save_best_only=True,
                             mode='min', save_weights_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4, min_lr=0.00001)
img_export = ProgressMonitor(generator=tb_gen, path='logs/unjpeg')
callbacks_list = [checkpoint, tb, reduce_lr, img_export]

model.fit_generator(train_gen, validation_data=val_gen, steps_per_epoch=200, epochs=100, validation_steps=20, callbacks=callbacks_list)
