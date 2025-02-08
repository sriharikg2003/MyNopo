import matplotlib.pyplot as plt
import numpy as np
import random 
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

img = img_as_float(imread('/data2/badrinath/NoPoSplat/t0.png'))
orig = img
if img.ndim == 2:
    img = np.stack([img] * 3, axis=-1)  
segments_slic = slic(img, n_segments=100, compactness=10, sigma=1, start_label=1)

plt.imshow(mark_boundaries(img, segments_slic))
plt.savefig('super.png')

img = segments_slic
super_pixel_coordinates = dict()
min_index  = img.flatten().min()
max_index = img.flatten().max()
for i in range(min_index , max_index+1):
    super_pixel_coordinates[i]=[]
for i in range(len(img)):
    for j in range(len(img[0])):
        super_pixel_coordinates[img[i][j]].append((i,j))
representation_gaussians = []





selected_superpixels = random.sample(list(super_pixel_coordinates.keys()), 2)
for sp in selected_superpixels:
    for coord in super_pixel_coordinates[sp]:
        x, y = coord
        orig[x, y, :] = 0





plt.imshow(mark_boundaries(orig, segments_slic))
plt.savefig('super_black.png')


percentage = 10 
for i in selected_superpixels:
    num_pixels = int(len(super_pixel_coordinates[i]) * percentage / 100)
    representation_gaussians.extend(random.sample(super_pixel_coordinates[i], num_pixels))


po = orig*0

for i in representation_gaussians:
    x,y = i
    po[x,y,:] = 1

plt.imshow(mark_boundaries(po, segments_slic))
plt.savefig('gauss.png')