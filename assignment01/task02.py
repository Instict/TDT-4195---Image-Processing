import numpy as np
import matplotlib.pyplot as plt

from scipy import misc
from PIL import Image


#	load an image an display it using matplotlib
image = misc.imread('./lochness.tiff')
plt.figure(1)
plt.title('Using matplotlib')
plt.imshow(image, interpolation='nearest')
plt.show()

#	loaded using NumPy array using PIL
img = Image.open('./lochness.tiff')
img = np.array(img)
print(np.array_equal(image, img))
plt.figure(2)
plt.title('Using NumPy array using PIL')
plt.imshow(img, interpolation='nearest')
plt.show()

#	transform image using the natural logarithm
imagelog = image.astype(np.float32)
c = 255 / np.log(np.max(imagelog) + 1)  # Scaling
imagelog = c * np.log(imagelog + 1)
imagelog = imagelog.astype(np.uint8)
plt.figure()
plt.imshow(imagelog)
plt.axis('off')  # Turn of axis numbers and ticks
plt.show()

#	Plot each channel on its own using subplots
r = image[..., 0]  # Equivalent to image[:, :, 0]
g = image[..., 1]
b = image[..., 2]

_, ax = plt.subplots(1, 3, figsize=(10, 8))
ax[0].imshow(r, cmap=plt.cm.Reds)  # Red colour map
ax[0].set_axis_off()
ax[1].imshow(g, cmap=plt.cm.Greens)  # Green colour map
ax[1].set_axis_off()
ax[2].imshow(b, cmap=plt.cm.Blues)  # Blue colour map
ax[2].set_axis_off()
plt.show()