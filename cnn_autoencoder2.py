import glob
import os

import matplotlib.pyplot as plt
from keras.callbacks import RemoteMonitor, ModelCheckpoint

from image_patches import batch_generator, stream_patches
from models import DeepDenoiseSR
from progress_monitor import ProgressMonitor

img_width = img_height = 32
img_depth = 3
image_dim = (img_width, img_height, img_depth)
scale = 1
batch_size = 128


def build_model(model_dir):
    model = DeepDenoiseSR(image_dim)
    files = glob.glob(model_dir + '/*.hdf5')
    if len(files):
        files.sort(key=os.path.getmtime, reverse=True)
        print('Loading model ' + files[0])
        if os.path.isfile(files[0]):
            print('Resuming training')
            model.load_weights(files[0])
    return model


autoencoder = build_model('model')


filepath = "model/model-epoch-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_PSNRLoss', save_best_only=True,
                                                   mode='max', save_weights_only=True, verbose=1)
remote = RemoteMonitor(root='http://localhost:9000')

# test_generator = batch_generator(data_dir='../../data2/test', dim=(img_width, img_height), max_patches=500, batch_size=128)
# val_generator = batch_generator(data_dir='../../data2/test', dim=(img_width, img_height), max_patches=500, batch_size=128)
#generator = batch_generator(data_dir='../../data2/train', dim=(img_width, img_height), max_patches=5000, batch_size=batch_size)
generator = stream_patches(data_dir='../../data2/patches/train', batch_size=batch_size)
val_generator = stream_patches(data_dir='../../data2/patches/test', batch_size=batch_size)
test_generator = stream_patches(data_dir='../../data2/patches/test', batch_size=batch_size)

progress = ProgressMonitor(generator=test_generator, dim=image_dim)
callbacks_list = [checkpoint, remote, progress]

hist = autoencoder.fit_generator(generator, samples_per_epoch=batch_size*500, nb_epoch=30, callbacks=callbacks_list, validation_data=val_generator, nb_val_samples=batch_size*100)

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
