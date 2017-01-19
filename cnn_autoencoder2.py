import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard

from image_patches import stream_patches_live
from models import SRCNN
from progress_monitor import ProgressMonitor

img_width = img_height = 32
img_depth = 3
image_dim = (img_width, img_height, img_depth)
scale = 1
batch_size = 128


def build_model(model_dir, type):
    model = type(image_dim)
    files = glob.glob(model_dir + '/*.hdf5')
    if len(files):
        files.sort(key=os.path.getmtime, reverse=True)
        print('Loading model ' + files[0])
        if os.path.isfile(files[0]):
            print('Resuming training')
            model.load_weights(files[0])
    return model


autoencoder = build_model('model', SRCNN)

# filepath = "model/model-epoch-{epoch:02d}-{loss:.4f}.hdf5"
filepath = "model/srcnn-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True,
                             mode='min', save_weights_only=True, verbose=1)


# remote = RemoteMonitor(root='http://localhost:9000')


def gen_noise(x_image):
    width, height, channels = x_image.shape
    x_image = x_image[:, :, 0:channels] * np.asarray(np.random.rand(height, width, 1) > 0.07, dtype='float32')
    np.place(x_image, x_image == 0., 1)
    return x_image


generator = stream_patches_live(data_dir='../../Pictures/people/train', dim=(32, 32), batch_size=batch_size,
                                noise_fn=gen_noise)
val_generator = stream_patches_live(data_dir='../../Pictures/people/test', dim=(32, 32), batch_size=batch_size,
                                    noise_fn=gen_noise)
test_generator = stream_patches_live(data_dir='../../Pictures/people/test', dim=(32, 32), batch_size=batch_size,
                                     noise_fn=gen_noise)

progress = ProgressMonitor(generator=test_generator, path='./logs', dim=image_dim)
tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
callbacks_list = [checkpoint, progress, tb]

hist = autoencoder.fit_generator(generator, samples_per_epoch=batch_size * 500, nb_epoch=100, callbacks=callbacks_list,
                                 validation_data=val_generator, nb_val_samples=batch_size * 100)

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
