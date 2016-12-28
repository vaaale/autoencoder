import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np
import matplotlib.image as mpimg
from models import DeepDenoiseSR, SRCNN
from scipy.ndimage.filters import gaussian_filter

p_dim = (32, 32, 3)

print('Building model...')
model = DeepDenoiseSR(p_dim)
print('Loadin weights...')
model.load_weights('model/backup/DeepDenoise2-32x32-0.0008.hdf5')

face = mpimg.imread('../../data2/test/frames-00002.jpg')
orig = face
#face = imresize(face, 400)
print('Image shape')
print(face.shape)

patches = []
m_dim = int(face.shape[0] - p_dim[0])
n_dim = int(face.shape[1] - p_dim[1])
for m in np.arange(0, m_dim, p_dim[0]):
    for n in np.arange(0, n_dim, p_dim[1]):
        patch_x = face[m:m + p_dim[1], n:n + p_dim[0]]
        input = patch_x.reshape((1, *p_dim)) / 255.
        patches.append(input)

arr = np.concatenate(patches, axis=0)
width = (np.ceil(n_dim/32))
height = (np.ceil(m_dim/32))
result = model.predict(arr, batch_size=512)
output = np.zeros((height*32, width*32, 3))
for m in np.arange(height):
    for n in np.arange(width):
        output[m*32:m*32 + 32, n*32:n*32 + 32] = result[m*width + n]


print('Displaying result.')
plt.figure(figsize=(20, 4))
ax = plt.subplot(1, 3, 1)
plt.imshow(orig)

ax = plt.subplot(1, 3, 2)
plt.imshow(face)

ax = plt.subplot(1, 3, 3)
plt.imshow(output)
plt.show()
